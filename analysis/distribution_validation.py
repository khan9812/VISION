"""
Distribution Validation - Synthetic Data Testing
=================================================
Validates Dispersion Index (DI) metric using 4 synthetic test cases:
- Case A: Same size + uniform distribution
- Case B: Same size + clustered distribution
- Case C: Variable size + uniform distribution
- Case D: Variable size + clustered distribution

Analyzes:
- CV and DI calculation accuracy
- Particle size confounding effect (Case C correlation test)
- Visual comparison with ground truth

Uses panalysis_debug.py logic:
- Concave hull (alpha shape) boundary detection
- Unified Voronoi areas with boundary pixel consolidation
- Full-image mask removal
- IQR outlier filtering
- BM3D denoising preprocessing

Author: Claude
Date: 2025-11-27
"""
import os
import sys
import gc
# Add parent directory to path for modules access
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)


import os
import sys
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.stats import pearsonr, iqr
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageDraw
import cv2
import time
import torch
import glob
from modules.pipeline_cache import load_pipeline_cache, save_pipeline_cache
from modules.project_paths import get_validation_dir

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Force GPU memory cleanup before starting (important when running after other scripts)
if torch.cuda.is_available():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()
    # Print initial memory status
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"[INFO] Initial GPU memory: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved")

# SAM2 imports
sys.path.append("..")
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Import modules from panalysis_debug.py
from modules.preprocessing import auto_preprocess, extract_boundary_pixels_dict
# MAD noise estimation removed - using fixed sigma=40 for BM3D
from modules.visualization import (
    find_optimal_alpha, calculate_polygon_area,
    filter_masks_based_on_centroid_containment,
    visualize_and_get_centroids
)
from modules.distribution_analysis import (
    compute_unified_voronoi_areas, calculate_distribution_metrics
)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {device}")

# Output - use absolute path from project root
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
OUTPUT_DIR = os.environ.get(
    "DISTRIBUTION_VALIDATION_OUTPUT_DIR",
    str(get_validation_dir("distribution_validation")),
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Legacy synthetic-case labels are remapped while preserving the published A-E
# labels. New case F is the variable-size counterpart of perfect-uniform case A.
CASE_LABEL_REMAP = {
    "E": "A",
    "A": "B",
    "C": "C",
    "B": "D",
    "D": "E",
    "F": "F",
}
CASE_ORDER_ALPHABET = ["A", "B", "C", "D", "E", "F"]
THEORETICAL_CASE_ORDER = ["A", "F", "B", "C", "D", "E"]
CASE_NAME_REMAP_PATTERN = re.compile(r"^Case_([A-F])(?=_|$)", re.IGNORECASE)
THEORETICAL_ORDER_TEXT = "A ~= F > B >= C > D >= E"
PAIRWISE_DIFF_PAIRS = [("A", "F"), ("B", "C"), ("D", "E")]
# The synthetic generators and historical cache keep the internal labels above.
# Every user-facing workbook and figure is written in this manuscript order.
MANUSCRIPT_CASE_REMAP = {
    "A": "A",
    "F": "B",
    "B": "C",
    "C": "D",
    "D": "E",
    "E": "F",
}
MANUSCRIPT_CASE_REMAP_REVERSE = {
    manuscript_case: code_case
    for code_case, manuscript_case in MANUSCRIPT_CASE_REMAP.items()
}
MANUSCRIPT_CASE_ORDER = ["A", "B", "C", "D", "E", "F"]
MANUSCRIPT_THEORETICAL_ORDER_TEXT = "A ~= B > C >= D > E >= F"
MANUSCRIPT_LABEL_SCHEME = "manuscript_A_to_F_v1"
MANUSCRIPT_PAIR_REMAP = {
    f"{left}-{right}": (
        f"{MANUSCRIPT_CASE_REMAP[left]}-{MANUSCRIPT_CASE_REMAP[right]}"
    )
    for left, right in PAIRWISE_DIFF_PAIRS
}
POSITIONAL_ORDER_PAIRS = [
    ("A", "B"),
    ("B", "D"),
    ("A", "D"),
    ("F", "C"),
    ("C", "E"),
    ("F", "E"),
]
CONDITION_DEFINITIONS = {
    "A": "perfectly regular positions + identical particle size",
    "F": "perfectly regular positions + different particle size",
    "B": "mildly perturbed positions + identical particle size",
    "C": "mildly perturbed positions + different particle size",
    "D": "clustered positions + identical particle size",
    "E": "clustered positions + different particle size",
}

print("="*80)
print("DISTRIBUTION VALIDATION - SYNTHETIC DATA")
print("="*80)


# ============================================================================
# SYNTHETIC DATA GENERATION WITH BM3D PREPROCESSING
# ============================================================================

def generate_synthetic_tem_image(particle_positions, particle_sizes,
                                  image_size=(1024, 1024),
                                  pixel_to_nm=2.0,
                                  apply_bm3d=False):
    """
    Generate synthetic TEM-like image with particles and BM3D preprocessing.
    
    Args:
        particle_positions: List of (x, y) tuples in nm
        particle_sizes: List of diameters in nm
        image_size: (height, width) in pixels
        pixel_to_nm: Conversion factor (nm per pixel)
        apply_bm3d: Whether to apply BM3D denoising
    
    Returns:
        PIL Image object (grayscale), actual positions in pixels, boundary_pixels dict
    """
    # Create base image
    img = Image.new('L', image_size, color=255)  # White background
    draw = ImageDraw.Draw(img)
    
    positions_px = []
    
    for (x_nm, y_nm), size_nm in zip(particle_positions, particle_sizes):
        # Convert nm to pixels
        x_px = int(x_nm / pixel_to_nm)
        y_px = int(y_nm / pixel_to_nm)
        radius_px = int((size_nm / 2) / pixel_to_nm)
        
        # Draw black circle (particle)
        bbox = [x_px - radius_px, y_px - radius_px,
                x_px + radius_px, y_px + radius_px]
        draw.ellipse(bbox, fill=0, outline=0)
        
        positions_px.append((x_px, y_px))
    
    # Convert to numpy array (BGR for preprocessing)
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    
    # Apply BM3D preprocessing with fixed sigma=40
    if apply_bm3d:
        print("  [BM3D] Applying denoising with fixed sigma=40...")
        img_bgr, preproc_info = auto_preprocess(img_bgr, sigma_psd=40, force_noise2sr=True)
        print(f"  [BM3D] ?=40.00, method={preproc_info['denoising_method']}")
    
    return img_bgr, positions_px


def nm_to_px(value_nm: float, pixel_to_nm: float) -> float:
    return value_nm / pixel_to_nm


def calculate_cluster_radius(particles_per_cluster: int, max_particle_radius_nm: float) -> float:
    """
    Calculate cluster radius to guarantee all particles fit without overlap.

    Uses hexagonal packing efficiency (~0.9069) for optimal space utilization.

    Args:
        particles_per_cluster: Number of particles in cluster
        max_particle_radius_nm: Maximum particle radius in nm

    Returns:
        cluster_radius_nm: Minimum cluster radius to fit all particles
    """
    # Hexagonal packing efficiency factor
    packing_efficiency = 0.9069
    safety_factor = 1.5  # Increased from 1.3 to prevent overlap

    # Required area for all particles (with safety margin)
    total_particle_area = particles_per_cluster * np.pi * (max_particle_radius_nm ** 2)
    cluster_area = total_particle_area / packing_efficiency * safety_factor

    # Calculate cluster radius from area
    cluster_radius_nm = np.sqrt(cluster_area / np.pi)

    return cluster_radius_nm


def ensure_non_overlap_cluster(centers_nm, radii_nm, new_center_nm, new_radius_nm, safety: float = 1.05, min_gap_nm: float = 5.0) -> bool:
    """
    Check if new circle (nm) does not overlap existing circles.
    
    Args:
        min_gap_nm: Minimum gap between particle edges (nm), default 5nm
    """
    for (cx, cy), r in zip(centers_nm, radii_nm):
        dx = new_center_nm[0] - cx
        dy = new_center_nm[1] - cy
        dist = (dx*dx + dy*dy) ** 0.5
        # Required distance = sum of radii * safety + minimum gap
        min_dist = (r + new_radius_nm) * safety + min_gap_nm
        if dist < min_dist:
            return False
    return True


def ensure_non_overlap_global(all_positions_nm, all_radii_nm, new_center_nm, new_radius_nm, safety: float = 1.05, min_gap_nm: float = 5.0) -> bool:
    """
    Check if new circle (nm) does not overlap with ANY existing circles globally.
    
    Args:
        all_positions_nm: List of all previously placed positions (all clusters)
        all_radii_nm: List of all previously placed radii (all clusters)
        new_center_nm: New particle center (x, y) in nm
        new_radius_nm: New particle radius in nm
        safety: Safety factor for minimum distance (1.05 = 5% extra margin)
        min_gap_nm: Minimum gap between particle edges (nm), default 5nm
    
    Returns:
        True if no overlap, False if overlap detected
    """
    for (cx, cy), r in zip(all_positions_nm, all_radii_nm):
        dx = new_center_nm[0] - cx
        dy = new_center_nm[1] - cy
        dist = (dx*dx + dy*dy) ** 0.5
        # Required distance = sum of radii * safety + minimum gap
        min_dist = (r + new_radius_nm) * safety + min_gap_nm
        if dist < min_dist:
            return False
    return True


def generate_clustered_positions_non_overlap(n_clusters: int,
                                             particles_per_cluster: int,
                                             cluster_radius_nm: float,
                                             image_size_nm: float,
                                             radii_nm: list,
                                             pixel_to_nm: float,
                                             max_attempts_per_point: int = 5000,
                                             safety: float = 1.05,
                                             min_gap_nm: float = 5.0,
                                             seed: int = 42,
                                             variable_size: bool = False,
                                             diameter_options: list = None):
    """
    Generate clustered positions with GUARANTEED particle generation (no loss).

    Automatically expands cluster_radius_nm if placement fails to ensure all particles fit.

    Args:
        radii_nm: list of particle radii (nm) for ALL particles (ignored if variable_size=True)
        cluster_radius_nm: Initial cluster radius (will auto-expand if needed)
        min_gap_nm: Minimum gap between particle edges (nm), default 5nm
        variable_size: If True, randomly assign diameters from diameter_options
        diameter_options: List of possible diameters (nm) for variable size mode

    Returns:
        positions_nm: List of (x_nm, y_nm) with len == len(radii_nm) (GUARANTEED)
        assigned_radii_nm: List of radii used
    """
    rng = np.random.default_rng(seed)
    positions_nm = []
    assigned_radii_nm = []

    # Ensure radii match total particles
    total_particles = n_clusters * particles_per_cluster

    if variable_size and diameter_options:
        # Variable size mode: randomly assign diameters
        diameters_nm = rng.choice(diameter_options, size=total_particles).tolist()
        radii_nm = [d / 2.0 for d in diameters_nm]
    else:
        # Fixed size mode: use provided radii
        if len(radii_nm) != total_particles:
            if len(radii_nm) < total_particles:
                repeats = (total_particles + len(radii_nm) - 1) // len(radii_nm)
                radii_nm = (radii_nm * repeats)[:total_particles]
            else:
                radii_nm = radii_nm[:total_particles]

    # Calculate optimal cluster radius if needed
    max_r_nm = max(radii_nm) if len(radii_nm) else 0
    optimal_radius = calculate_cluster_radius(particles_per_cluster, max_r_nm)
    cluster_radius_nm = max(cluster_radius_nm, optimal_radius)

    # Use GRID-based cluster placement (guaranteed fit for 256횞256)
    margin = 50.0  # Reduced margin for 256횞256
    cluster_centers = generate_cluster_grid_positions(n_clusters, margin, image_size_nm)

    radius_idx = 0

    # Process each cluster with auto-expansion if needed
    for ci, (ccx, ccy) in enumerate(cluster_centers):
        local_centers = []
        local_radii = []
        placed = 0
        expansion_count = 0
        current_cluster_radius = cluster_radius_nm

        # Place all particles for this cluster (guaranteed)
        while placed < particles_per_cluster and radius_idx < len(radii_nm):
            r_nm = radii_nm[radius_idx]

            # Try to place particle
            success = False
            for attempt in range(max_attempts_per_point):
                ang = rng.uniform(0, 2*np.pi)
                rho = rng.uniform(0, max(1e-6, current_cluster_radius - r_nm))
                x_nm = ccx + rho * np.cos(ang)
                y_nm = ccy + rho * np.sin(ang)

                # Check inside cluster circle
                if ((x_nm - ccx)**2 + (y_nm - ccy)**2) ** 0.5 > (current_cluster_radius - r_nm):
                    continue

                # Check non-overlap within this cluster (local check)
                if not ensure_non_overlap_cluster(local_centers, local_radii, (x_nm, y_nm), r_nm, safety=safety, min_gap_nm=min_gap_nm):
                    continue
                
                # Check non-overlap with ALL previously placed particles (global check)
                if not ensure_non_overlap_global(positions_nm, assigned_radii_nm, (x_nm, y_nm), r_nm, safety=safety, min_gap_nm=min_gap_nm):
                    continue
                
                # Passed both checks - place particle
                local_centers.append((x_nm, y_nm))
                local_radii.append(r_nm)
                radius_idx += 1
                placed += 1
                success = True
                break

            # If failed, expand cluster radius and retry THIS particle
            if not success:
                expansion_count += 1
                current_cluster_radius *= 1.2  # 20% expansion
                if expansion_count == 1:
                    print(f"  [AUTO-EXPAND] Cluster {ci+1}: Expanding radius to {current_cluster_radius:.1f}nm")

        # Commit local placements
        positions_nm.extend(local_centers)
        assigned_radii_nm.extend(local_radii)

        if expansion_count > 0:
            print(f"  [CLUSTER {ci+1}] Placed {placed}/{particles_per_cluster} (expanded {expansion_count}x)")

    # Verify no particle loss
    expected_total = n_clusters * particles_per_cluster
    actual_total = len(positions_nm)

    if actual_total != expected_total:
        raise RuntimeError(
            f"PARTICLE GENERATION FAILED: Expected {expected_total}, got {actual_total}. "
            "This should not happen with auto-expansion enabled."
        )

    return positions_nm, assigned_radii_nm


def generate_cluster_grid_positions(n_clusters, margin, image_size_nm):
    """
    Generate grid-based cluster center positions (GUARANTEED fit for 512횞512).

    Args:
        n_clusters: Number of clusters (4, 5, or 6)
        margin: Margin from edges (nm)
        image_size_nm: Image size in nm

    Returns:
        List of (x, y) cluster center positions in nm
    """
    import math

    # Calculate available space
    available = image_size_nm - 2 * margin

    # Determine grid layout based on cluster count
    if n_clusters == 4:
        grid_rows, grid_cols = 2, 2
    elif n_clusters == 5:
        # 2횞2 grid + 1 center position
        grid_rows, grid_cols = 2, 2
    elif n_clusters == 6:
        grid_rows, grid_cols = 2, 3
    else:
        # Fallback: square-ish grid
        grid_cols = math.ceil(math.sqrt(n_clusters))
        grid_rows = math.ceil(n_clusters / grid_cols)

    # Calculate spacing
    h_spacing = available / (grid_cols + 1)
    v_spacing = available / (grid_rows + 1)

    positions = []
    for i in range(grid_rows):
        for j in range(grid_cols):
            if len(positions) >= n_clusters:
                break
            x = margin + (j + 1) * h_spacing
            y = margin + (i + 1) * v_spacing
            positions.append((x, y))
        if len(positions) >= n_clusters:
            break

    # Special case for 5 clusters: add center position
    if n_clusters == 5 and len(positions) == 4:
        center_x = image_size_nm / 2
        center_y = image_size_nm / 2
        positions.append((center_x, center_y))

    return positions


def generate_grid_positions(n_rows, n_cols, spacing_nm, margin_nm=200,
                            variable_size=False, diameter_options=None, seed=42):
    """
    Generate uniform grid positions with optional variable sizes.

    Args:
        n_rows, n_cols: Grid dimensions
        spacing_nm: Distance between particles in nm
        margin_nm: Margin from edges
        variable_size: If True, assign random diameters from diameter_options
        diameter_options: List of possible diameters (nm) for variable size mode
        seed: Random seed for reproducibility

    Returns:
        positions: List of (x, y) positions in nm
        diameters: List of diameters (nm) for each position (if variable_size=True)
    """
    positions = []
    for i in range(n_rows):
        for j in range(n_cols):
            x = margin_nm + j * spacing_nm
            y = margin_nm + i * spacing_nm
            positions.append((x, y))

    if variable_size and diameter_options:
        rng = np.random.default_rng(seed)
        diameters = rng.choice(diameter_options, size=len(positions)).tolist()
        return positions, diameters
    else:
        return positions


def generate_clustered_positions(n_clusters, particles_per_cluster,
                                  cluster_radius_nm, image_size_nm):
    """
    Generate clustered particle positions.
    
    Args:
        n_clusters: Number of clusters
        particles_per_cluster: Particles in each cluster
        cluster_radius_nm: Cluster size in nm
        image_size_nm: Image dimensions in nm (assume square)
    
    Returns:
        List of (x, y) positions in nm
    """
    positions = []
    np.random.seed(42)  # Reproducibility
    
    # Random cluster centers
    cluster_centers = []
    margin = cluster_radius_nm + 100
    for _ in range(n_clusters):
        cx = np.random.uniform(margin, image_size_nm - margin)
        cy = np.random.uniform(margin, image_size_nm - margin)
        cluster_centers.append((cx, cy))
    
    # Generate particles within each cluster
    for cx, cy in cluster_centers:
        for _ in range(particles_per_cluster):
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, cluster_radius_nm)
            x = cx + radius * np.cos(angle)
            y = cy + radius * np.sin(angle)
            positions.append((x, y))
    
    return positions


# ============================================================================
# SAM2 MODEL LOADING
# ============================================================================

def get_sam2_config_path(config_name):
    """Get absolute path to SAM2 config from installed package"""
    import sam2
    sam2_path = os.path.dirname(sam2.__file__)
    return os.path.join(sam2_path, 'configs', 'sam2.1', config_name)


def load_sam2_model():
    """Load SAM2 model for synthetic data analysis"""
    # Get project root (parent of code/ directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    model_type = "hiera_l"
    sam_checkpoint = os.path.join(project_root, "checkpoints", "sam2.1_hiera_large.pt")
    sam_config = get_sam2_config_path("sam2.1_hiera_l.yaml")

    if not os.path.exists(sam_checkpoint):
        raise FileNotFoundError(f"SAM2 checkpoint not found: {sam_checkpoint}")

    print(f"  [SAM2] Loading {model_type}...")
    sam = build_sam2(sam_config, sam_checkpoint, device=device)
    return sam, model_type


# ============================================================================
# SAM2 MASK GENERATION WITH FILTERING
# ============================================================================

def remove_overlapped_masks(masks, iou_threshold=0.5):
    """
    Remove overlapped masks based on IoU.
    Keep masks with higher predicted IoU scores.
    """
    if len(masks) == 0:
        return masks

    sorted_masks = sorted(masks, key=lambda x: x.get('predicted_iou', 0), reverse=True)
    keep_masks = []

    for current_mask in sorted_masks:
        current_seg = current_mask['segmentation']
        should_keep = True

        for kept_mask in keep_masks:
            kept_seg = kept_mask['segmentation']
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

def generate_and_filter_masks(image_bgr, sam_model):
    """
    Generate SAM2 masks and apply filtering pipeline.

    Filters:
    1. Overlap removal (IoU > 0.5)
    2. Background removal (4-border touching OR >90% area)
    3. IQR outlier filtering
    """
    start_time = time.time()
    print("\n  [SAM2] Generating masks...")

    # Clear GPU cache before SAM2 generation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # UNIFIED SAM2 PARAMETERS (consistent across all validation scripts)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam_model,
        points_per_side=32,                    # Standard grid density
        points_per_batch=256,                  # Batch size for point processing
        pred_iou_thresh=0.95,                  # Empirical default
        stability_score_thresh=0.80,           # Empirical default (unified)
        crop_n_layers=1,                       # Multi-scale cropping
        crop_n_points_downscale_factor=2,      # Point density in crops
        crop_nms_thresh=0.7,                   # NMS threshold for crops
        box_nms_thresh=0.7,                    # NMS threshold for boxes
        use_m2m=True                           # Mask-to-mask refinement
    )

    masks = mask_generator.generate(image_bgr)
    elapsed = time.time() - start_time
    print(f"  [SAM2] Generated {len(masks)} masks ({elapsed:.1f}s)")

    # Clear GPU cache after SAM2 generation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Filter 1: Remove overlapped masks
    masks_before = len(masks)
    masks = remove_overlapped_masks(masks, iou_threshold=0.5)
    print(f"  [FILTER 1] Overlap removal: {masks_before} ??{len(masks)}")

    # Filter 2: Remove background masks
    total_pixels = image_bgr.shape[0] * image_bgr.shape[1]
    border_tolerance = 5
    masks_before = len(masks)
    filtered_masks = []

    for m in masks:
        bbox = m['bbox']
        x, y, w, h = bbox
        coverage = (w * h) / total_pixels

        # Skip if >90% coverage
        if coverage > 0.90:
            continue

        # Skip if touches all 4 borders
        touches_all = (x <= border_tolerance and y <= border_tolerance and
                      (x + w) >= (image_bgr.shape[1] - border_tolerance) and
                      (y + h) >= (image_bgr.shape[0] - border_tolerance))
        if touches_all:
            continue

        filtered_masks.append(m)

    masks = filtered_masks
    print(f"  [FILTER 2] Background removal: {masks_before} ??{len(masks)}")

    # Filter 3: IQR outlier filtering - DISABLED for synthetic images
    # Synthetic images have known particle sizes (15, 20, 25nm), no outliers expected
    # IQR filtering was incorrectly removing valid particles with variable sizes
    print(f"  [FILTER 3] IQR filtering: DISABLED (synthetic data with known sizes)")

    return masks


# ============================================================================
# UNIFIED VORONOI ANALYSIS (panalysis_debug.py logic)
# ============================================================================

def analyze_voronoi_with_concave_hull(masks, image_bgr):
    """
    Compute Voronoi areas using the shared convex analysis hull.

    The function name is retained for compatibility with existing validation
    scripts and cached result readers.

    Follows panalysis_debug.py workflow:
    1. Extract centroids and boundary pixels
    2. Build the convex hull and apply the shared inward centroid filter
    3. Compute unified Voronoi areas with hull intersection

    Returns:
        voronoi_areas_list: List of Voronoi areas
        inside_centroids: Centroids inside hull (used for Voronoi)
        filtered_centroids: All detected centroids (before hull filtering)
        concave_vertex: Hull vertices
    """
    print("\n  [VORONOI] Extracting centroids and boundaries...")

    # Extract centroids - returns only valid centroids (some masks may be skipped)
    _, centroids = visualize_and_get_centroids(image_bgr, masks)
    
    # CRITICAL FIX: Sync masks with centroids
    # visualize_and_get_centroids may skip masks with invalid contours
    # We need to filter masks to match centroids length
    if len(centroids) != len(masks):
        print(f"  [SYNC] Masks ({len(masks)}) != Centroids ({len(centroids)}), re-syncing...")
        # Re-extract with mask filtering
        valid_masks = []
        valid_centroids = []
        for idx, mask_info in enumerate(masks):
            mask = mask_info['segmentation']
            binary_mask = (mask > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                continue
            
            valid_contours = [c for c in contours if cv2.contourArea(c) > 0]
            if not valid_contours:
                continue
            
            smallest_contour = min(valid_contours, key=cv2.contourArea)
            M = cv2.moments(smallest_contour)
            if M["m00"] != 0:
                centroid_x = int(M["m10"] / M["m00"])
                centroid_y = int(M["m01"] / M["m00"])
                valid_masks.append(mask_info)
                valid_centroids.append((centroid_x, centroid_y))
        
        masks = valid_masks
        centroids = valid_centroids
        print(f"  [SYNC] After sync: {len(masks)} masks, {len(centroids)} centroids")

    # Filter based on centroid containment
    filtered_masks, filtered_centroids = filter_masks_based_on_centroid_containment(
        masks, centroids
    )

    print(f"  [VORONOI] Centroids: {len(filtered_centroids)}")

    # Extract boundary pixels
    boundary_pixels = extract_boundary_pixels_dict(filtered_masks)
    print(f"  [VORONOI] Boundary pixels extracted for {len(boundary_pixels)} particles")

    # Build the shared convex analysis hull (legacy helper name retained).
    print(f"  [CONCAVE HULL] Finding optimal alpha...")
    inside_centroids, polygons, concave_vertex, alpha = find_optimal_alpha(
        filtered_centroids
    )

    if polygons is None:
        print("  [WARN] Failed to find convex analysis hull, using all centroids")
        inside_centroids = filtered_centroids
        concave_vertex = None
    else:
        excluded_count = len(filtered_centroids) - len(inside_centroids)
        print(f"  [CONCAVE HULL] 慣={alpha:.4f}, inside={len(inside_centroids)}/{len(filtered_centroids)} (excluded={excluded_count})")

    # Compute unified Voronoi areas
    print(f"  [VORONOI] Computing unified areas for {len(boundary_pixels)} particles...")
    voronoi_areas_dict, vor_obj, unified_regions = compute_unified_voronoi_areas(
        boundary_pixels, concave_vertex, inside_centroids
    )
    print(f"  [VORONOI] ??Completed")

    voronoi_areas_list = list(voronoi_areas_dict.values())
    print(f"  [VORONOI] Computed {len(voronoi_areas_list)} unified Voronoi areas")

    return voronoi_areas_list, inside_centroids, filtered_centroids, concave_vertex, vor_obj, unified_regions


def calculate_cv_di(areas):
    """
    Calculate Coefficient of Variation and Dispersion Index.
    
    Args:
        areas: List of Voronoi cell areas
    
    Returns:
        cv, ui
    """
    if len(areas) == 0:
        return np.nan, np.nan
    
    mean_area = np.mean(areas)
    std_area = np.std(areas, ddof=1) if len(areas) > 1 else 0.0
    
    if mean_area == 0:
        return np.nan, np.nan
    
    cv = std_area / mean_area
    ui = 1 / (1 + cv)
    
    return cv, ui


# ============================================================================
# TEST CASE GENERATION
# ============================================================================

def generate_case_a(sam_model, n_repeats=1):
    """Case A: Same size + uniform distribution with jitter (9 versions 횞 n_repeats images)
    NOTE: Saves positions for Case C to use (isolate size effects)"""
    print(f"\n[CASE A] Same size + uniform distribution with jitter (9 versions 횞 {n_repeats} repeats)")
    print("  [POSITION SAVING] Saving positions for Case C to reuse")

    all_variants = []
    all_results = []  # Accumulate ALL results for final saving
    diameters_nm = [23, 25, 27]  # Reduced for 256횞256 image
    grid_configs = [
        (5, 5),   # 25 particles
        (6, 6),    # 36 particles
        (7, 7)    # 49 particles
    ]
    margin_nm = 50  # Reduced margin for smaller image
    base_spacing_nm = 80  # Increased to prevent overlap (was 60)
    pixel_to_nm = 2.0
    image_size_px = 256  # Reduced from 512

    # Create directory for position files
    pos_dir = Path("../validation_positions")
    pos_dir.mkdir(exist_ok=True)

    vi = 0
    for d_nm in diameters_nm:
        for n_rows, n_cols in grid_configs:
            vi += 1
            n_particles = n_rows * n_cols
            print(f"\n  ?봽 Version {vi}/9 (Size={d_nm}nm, N={n_particles}, Grid={n_rows}횞{n_cols})...")

            for rep in range(n_repeats):
                seed = 1000 + vi * 100 + rep
                rng = np.random.default_rng(seed)

                # Calculate spacing to fit in image
                image_size_nm = image_size_px * pixel_to_nm  # 512nm
                available_space = image_size_nm - 2 * margin_nm  # 412nm
                max_spacing = available_space / max(n_rows - 1, n_cols - 1)  # Fit grid
                desired_spacing = max(base_spacing_nm, d_nm * 1.5)
                spacing_nm = min(desired_spacing, max_spacing)  # Don't exceed image

                positions = generate_grid_positions(n_rows, n_cols, spacing_nm, margin_nm=margin_nm)
                jitter = rng.uniform(-spacing_nm * 0.1, spacing_nm * 0.1, (len(positions), 2))
                positions = [(p[0] + j[0], p[1] + j[1]) for p, j in zip(positions, jitter)]

                # Save positions for Case C to reuse
                pos_file = pos_dir / f"case_A_v{vi}_r{rep+1}_positions.json"
                with open(pos_file, 'w') as f:
                    json.dump({'positions': positions, 'n_particles': n_particles}, f)

                sizes = [d_nm] * len(positions)
                img_bgr, positions_px = generate_synthetic_tem_image(
                    positions, sizes, image_size=(image_size_px, image_size_px), apply_bm3d=False
                )

                masks = generate_and_filter_masks(img_bgr, sam_model)
                voronoi_areas, centroids, all_centroids, hull, vor_obj, unified_regions = analyze_voronoi_with_concave_hull(masks, img_bgr)

                # Detection success monitoring
                detected = len(masks)
                success_rate = detected / n_particles * 100
                if rep == 0:
                    print(f"    [A{vi}] Detection: {detected}/{n_particles} ({success_rate:.1f}%)")

                variant_data = {
                    'name': f'Case_A_v{vi}_r{rep+1}',
                    'case': 'A',
                    'version': vi,
                    'repeat': rep + 1,
                    'image': img_bgr,
                    'positions_px': positions_px,
                    'sizes_nm': sizes,
                    'voronoi_areas': voronoi_areas,
                    'centroids': centroids,
                    'all_centroids': all_centroids,
                    'hull': hull,
                    'vor_obj': vor_obj,
                    'unified_regions': unified_regions,
                    'expected_di': '>0.95',
                    'description': f'Uniform grid, {d_nm}nm, N={n_particles}'
                }
                all_variants.append(variant_data)

                # Analyze and save visualization (accumulate results)
                result = analyze_case(variant_data, save_visualization=True)
                if result:
                    all_results.append(result)

            print(f"    ??Version {vi}: {n_repeats} images completed")

    return all_variants, all_results


def generate_case_b(sam_model, n_repeats=1):
    """Case B: Same size + clustered distribution (9 versions 횞 n_repeats images)
    NOTE: Saves positions for Case D to use (isolate size effects)"""
    print(f"\n[CASE B] Same size + clustered distribution (9 versions 횞 {n_repeats} repeats)")
    print("  [POSITION SAVING] Saving positions for Case D to reuse")

    all_variants = []
    all_results = []  # Accumulate ALL results for final saving
    diameters_nm = [23, 25, 27]  # Reduced for 256횞256 image
    cluster_configs = [
        (5, 5),    # 25 particles
        (6, 6),    # 36 particles
        (7, 7)     # 49 particles
    ]
    pixel_to_nm = 2.0
    image_size_px = 256  # Reduced from 512
    image_size_nm = image_size_px * pixel_to_nm

    # Create directory for position files
    pos_dir = Path("../validation_positions")
    pos_dir.mkdir(exist_ok=True)

    vi = 0
    for d_nm in diameters_nm:
        for n_clusters, particles_per_cluster in cluster_configs:
            vi += 1
            n_particles = n_clusters * particles_per_cluster
            print(f"\n  ?봽 Version {vi}/9 (Size={d_nm}nm, N={n_particles}, Clusters={n_clusters}횞{particles_per_cluster})...")

            for rep in range(n_repeats):
                seed = 2000 + vi * 100 + rep

                r_nm = d_nm / 2.0
                radii_nm = [r_nm] * n_particles
                cluster_radius_nm = calculate_cluster_radius(particles_per_cluster, r_nm)

                positions_nm, radii_used_nm = generate_clustered_positions_non_overlap(
                    n_clusters, particles_per_cluster, cluster_radius_nm,
                    image_size_nm, radii_nm, pixel_to_nm, seed=seed
                )

                # Save positions for Case D to reuse
                pos_file = pos_dir / f"case_B_v{vi}_r{rep+1}_positions.json"
                with open(pos_file, 'w') as f:
                    json.dump({'positions': positions_nm, 'n_particles': n_particles}, f)

                sizes_nm = [ru * 2.0 for ru in radii_used_nm]
                img_bgr, positions_px = generate_synthetic_tem_image(
                    positions_nm, sizes_nm, image_size=(image_size_px, image_size_px), apply_bm3d=False
                )

                masks = generate_and_filter_masks(img_bgr, sam_model)
                voronoi_areas, centroids, all_centroids, hull, vor_obj, unified_regions = analyze_voronoi_with_concave_hull(masks, img_bgr)

                # Detection success monitoring
                detected = len(masks)
                success_rate = detected / n_particles * 100
                if rep == 0:
                    print(f"    [B{vi}] Detection: {detected}/{n_particles} ({success_rate:.1f}%)")

                variant_data = {
                    'name': f'Case_B_v{vi}_r{rep+1}',
                    'case': 'B',
                    'version': vi,
                    'repeat': rep + 1,
                    'image': img_bgr,
                    'positions_px': positions_px,
                    'sizes_nm': sizes_nm,
                    'voronoi_areas': voronoi_areas,
                    'centroids': centroids,
                    'all_centroids': all_centroids,
                    'hull': hull,
                    'vor_obj': vor_obj,
                    'unified_regions': unified_regions,
                    'expected_di': '<0.7',
                    'description': f'Clustered, {d_nm}nm, N={n_particles}'
                }
                all_variants.append(variant_data)

                # Analyze and save visualization (accumulate results)z
                result = analyze_case(variant_data, save_visualization=True)
                if result:
                    all_results.append(result)

            print(f"    ??Version {vi}: {n_repeats} images completed")

    return all_variants, all_results


def generate_case_c(sam_model, n_repeats=1):
    """Case C: VARIABLE size + uniform distribution (9 versions 횞 n_repeats images)
    NOTE: Loads IDENTICAL POSITIONS from Case A, only changes sizes"""
    print(f"\n[CASE C] Variable size + uniform distribution (9 versions 횞 {n_repeats} repeats)")
    print("  [POSITION LOADING] Loading positions from Case A, changing sizes only")

    all_variants = []
    all_results = []  # Accumulate ALL results for final saving
    diameter_options = [23, 25, 27]  # Variable size options matching Case A
    grid_configs = [
        (5, 5),   # 25 particles
        (6, 6),    # 36 particles
        (7, 7)    # 49 particles
    ]
    pixel_to_nm = 2.0
    image_size_px = 256  # Reduced from 512

    pos_dir = Path("../validation_positions")

    vi = 0
    # Generate 9 versions: 3 diameter configs 횞 3 grid configs
    for _ in diameter_options:  # Loop 3 times for diameter variety
        for n_rows, n_cols in grid_configs:
            vi += 1
            n_particles = n_rows * n_cols
            print(f"\n  ?봽 Version {vi}/9 (Variable size, N={n_particles}, Grid={n_rows}횞{n_cols})...")

            for rep in range(n_repeats):
                # Load positions from Case A (IDENTICAL positions)
                pos_file = pos_dir / f"case_A_v{vi}_r{rep+1}_positions.json"
                if not pos_file.exists():
                    raise FileNotFoundError(f"Position file not found: {pos_file}. Run Case A first!")

                with open(pos_file, 'r') as f:
                    pos_data = json.load(f)
                    positions = pos_data['positions']

                # Generate VARIABLE sizes for the SAME positions
                seed = 3000 + vi * 100 + rep  # Different seed for size generation only
                rng = np.random.default_rng(seed)
                sizes = rng.choice(diameter_options, size=len(positions)).tolist()

                img_bgr, positions_px = generate_synthetic_tem_image(
                    positions, sizes, image_size=(image_size_px, image_size_px), apply_bm3d=False
                )

                masks = generate_and_filter_masks(img_bgr, sam_model)
                voronoi_areas, centroids, all_centroids, hull, vor_obj, unified_regions = analyze_voronoi_with_concave_hull(masks, img_bgr)

                # Detection success monitoring
                detected = len(masks)
                success_rate = detected / n_particles * 100
                if rep == 0:
                    print(f"    [C{vi}] Detection: {detected}/{n_particles} ({success_rate:.1f}%)")

                variant_data = {
                    'name': f'Case_C_v{vi}_r{rep+1}',
                    'case': 'C',
                    'version': vi,
                    'repeat': rep + 1,
                    'image': img_bgr,
                    'positions_px': positions_px,
                    'sizes_nm': sizes,
                    'voronoi_areas': voronoi_areas,
                    'centroids': centroids,
                    'all_centroids': all_centroids,
                    'hull': hull,
                    'vor_obj': vor_obj,
                    'unified_regions': unified_regions,
                    'expected_di': '>0.90',
                    'description': f'Uniform grid, Variable size ({min(sizes)}-{max(sizes)}nm), N={n_particles}'
                }
                all_variants.append(variant_data)

                # Analyze and save visualization (accumulate results)
                result = analyze_case(variant_data, save_visualization=True)
                if result:
                    all_results.append(result)

            print(f"    ??Version {vi}: {n_repeats} images completed")

    return all_variants, all_results


def generate_case_d(sam_model, n_repeats=1):
    """Case D: VARIABLE size + clustered distribution (9 versions 횞 n_repeats images)
    NOTE: Loads IDENTICAL POSITIONS from Case B, only changes sizes"""
    print(f"\n[CASE D] Variable size + clustered distribution (9 versions 횞 {n_repeats} repeats)")
    print("  [POSITION LOADING] Loading positions from Case B, changing sizes only")

    all_variants = []
    all_results = []  # Accumulate ALL results for final saving
    diameter_options = [23, 25, 27]  # Variable size options matching Case B
    cluster_configs = [
        (5, 5),    # 25 particles
        (6, 6),    # 36 particles
        (7, 7)     # 49 particles
    ]
    pixel_to_nm = 2.0
    image_size_px = 256  # Reduced from 512

    pos_dir = Path("../validation_positions")

    vi = 0
    # Generate 9 versions: 3 diameter configs 횞 3 cluster configs
    for _ in diameter_options:  # Loop 3 times for diameter variety
        for n_clusters, particles_per_cluster in cluster_configs:
            vi += 1
            n_particles = n_clusters * particles_per_cluster
            print(f"\n  ?봽 Version {vi}/9 (Variable size, N={n_particles}, Clusters={n_clusters}횞{particles_per_cluster})...")

            for rep in range(n_repeats):
                # Load positions from Case B (IDENTICAL positions)
                pos_file = pos_dir / f"case_B_v{vi}_r{rep+1}_positions.json"
                if not pos_file.exists():
                    raise FileNotFoundError(f"Position file not found: {pos_file}. Run Case B first!")

                with open(pos_file, 'r') as f:
                    pos_data = json.load(f)
                    positions_nm = pos_data['positions']

                # Generate VARIABLE sizes for the SAME positions
                seed = 4000 + vi * 100 + rep  # Different seed for size generation only
                rng = np.random.default_rng(seed)
                sizes_nm = rng.choice(diameter_options, size=len(positions_nm)).tolist()
                img_bgr, positions_px = generate_synthetic_tem_image(
                    positions_nm, sizes_nm, image_size=(image_size_px, image_size_px), apply_bm3d=False
                )

                masks = generate_and_filter_masks(img_bgr, sam_model)
                voronoi_areas, centroids, all_centroids, hull, vor_obj, unified_regions = analyze_voronoi_with_concave_hull(masks, img_bgr)

                # Detection success monitoring
                detected = len(masks)
                success_rate = detected / n_particles * 100
                if rep == 0:
                    print(f"    [D{vi}] Detection: {detected}/{n_particles} ({success_rate:.1f}%)")

                variant_data = {
                    'name': f'Case_D_v{vi}_r{rep+1}',
                    'case': 'D',
                    'version': vi,
                    'repeat': rep + 1,
                    'image': img_bgr,
                    'positions_px': positions_px,
                    'sizes_nm': sizes_nm,
                    'voronoi_areas': voronoi_areas,
                    'centroids': centroids,
                    'all_centroids': all_centroids,
                    'hull': hull,
                    'vor_obj': vor_obj,
                    'unified_regions': unified_regions,
                    'expected_di': '<0.7',
                    'description': f'Clustered, Variable size ({min(sizes_nm)}-{max(sizes_nm)}nm), N={n_particles}'
                }
                all_variants.append(variant_data)

                # Analyze and save visualization (accumulate results)
                result = analyze_case(variant_data, save_visualization=True)
                if result:
                    all_results.append(result)

            print(f"    ??Version {vi}: {n_repeats} images completed")

    return all_variants, all_results


def generate_case_e(sam_model, n_repeats=1):
    """Case E: Same size + PERFECT uniform distribution (no jitter) (9 versions 횞 n_repeats images)"""
    print(f"\n[CASE E] Same size + PERFECT uniform distribution (no jitter) (9 versions 횞 {n_repeats} repeats)")

    all_variants = []
    all_results = []  # Accumulate ALL results for final saving
    diameters_nm = [23, 25, 27]  # Reduced for 256횞256 image
    grid_configs = [
        (5, 5),   # 25 particles
        (6, 6),    # 36 particles
        (7, 7)    # 49 particles
    ]
    margin_nm = 50  # Reduced margin for smaller image
    base_spacing_nm = 80  # Increased to prevent overlap (was 60)
    pixel_to_nm = 2.0
    image_size_px = 256  # Reduced from 512

    vi = 0
    for d_nm in diameters_nm:
        for n_rows, n_cols in grid_configs:
            vi += 1
            n_particles = n_rows * n_cols
            print(f"\n  ?봽 Version {vi}/9 (Size={d_nm}nm, N={n_particles}, Grid={n_rows}횞{n_cols})...")

            for rep in range(n_repeats):
                seed = 500 + vi * 100 + rep  # Different seed offset from Case A
                rng = np.random.default_rng(seed)

                # Calculate spacing to fit in image
                image_size_nm = image_size_px * pixel_to_nm  # 512nm
                available_space = image_size_nm - 2 * margin_nm  # 412nm
                max_spacing = available_space / max(n_rows - 1, n_cols - 1)  # Fit grid
                desired_spacing = max(base_spacing_nm, d_nm * 1.5)
                spacing_nm = min(desired_spacing, max_spacing)  # Don't exceed image

                positions = generate_grid_positions(n_rows, n_cols, spacing_nm, margin_nm=margin_nm)
                # NO JITTER - perfect uniformity

                sizes = [d_nm] * len(positions)
                img_bgr, positions_px = generate_synthetic_tem_image(
                    positions, sizes, image_size=(image_size_px, image_size_px), apply_bm3d=False
                )

                masks = generate_and_filter_masks(img_bgr, sam_model)
                voronoi_areas, centroids, all_centroids, hull, vor_obj, unified_regions = analyze_voronoi_with_concave_hull(masks, img_bgr)

                # Detection success monitoring
                detected = len(masks)
                success_rate = detected / n_particles * 100
                if rep == 0:
                    print(f"    [E{vi}] Detection: {detected}/{n_particles} ({success_rate:.1f}%)")

                variant_data = {
                    'name': f'Case_E_v{vi}_r{rep+1}',
                    'case': 'E',
                    'version': vi,
                    'repeat': rep + 1,
                    'image': img_bgr,
                    'positions_px': positions_px,
                    'sizes_nm': sizes,
                    'voronoi_areas': voronoi_areas,
                    'centroids': centroids,
                    'all_centroids': all_centroids,
                    'hull': hull,
                    'vor_obj': vor_obj,
                    'unified_regions': unified_regions,
                    'expected_di': '1.0',
                    'description': f'Perfect uniform grid (no jitter), {d_nm}nm, N={n_particles}'
                }
                all_variants.append(variant_data)

                # Analyze and save visualization (accumulate results)
                result = analyze_case(variant_data, save_visualization=True)
                if result:
                    all_results.append(result)

            print(f"    ??Version {vi}: {n_repeats} images completed")

    return all_variants, all_results


def generate_case_f(sam_model, n_repeats=1):
    """Case F: Variable size + perfectly uniform distribution (no jitter)."""
    print(
        f"\n[CASE F] Variable size + PERFECT uniform distribution "
        f"(no jitter) (9 versions x {n_repeats} repeats)"
    )
    print("  [CONTROL] Positions match Case E; size draws match Case C")

    all_variants = []
    all_results = []
    diameter_options = [23, 25, 27]
    grid_configs = [
        (5, 5),
        (6, 6),
        (7, 7),
    ]
    margin_nm = 50
    base_spacing_nm = 80
    pixel_to_nm = 2.0
    image_size_px = 256

    vi = 0
    for reference_diameter_nm in diameter_options:
        for n_rows, n_cols in grid_configs:
            vi += 1
            n_particles = n_rows * n_cols
            print(
                f"\n  Version {vi}/9 (Variable size, N={n_particles}, "
                f"Grid={n_rows}x{n_cols})..."
            )

            image_size_nm = image_size_px * pixel_to_nm
            available_space = image_size_nm - 2 * margin_nm
            max_spacing = available_space / max(n_rows - 1, n_cols - 1)
            desired_spacing = max(base_spacing_nm, reference_diameter_nm * 1.5)
            spacing_nm = min(desired_spacing, max_spacing)
            positions = generate_grid_positions(
                n_rows, n_cols, spacing_nm, margin_nm=margin_nm
            )

            for rep in range(n_repeats):
                # This is deliberately identical to Case C's size seed so each
                # version/repeat has the same particle count and diameter array.
                seed = 3000 + vi * 100 + rep
                rng = np.random.default_rng(seed)
                sizes = rng.choice(
                    diameter_options, size=len(positions)
                ).tolist()
                img_bgr, positions_px = generate_synthetic_tem_image(
                    positions,
                    sizes,
                    image_size=(image_size_px, image_size_px),
                    apply_bm3d=False,
                )

                masks = generate_and_filter_masks(img_bgr, sam_model)
                (
                    voronoi_areas,
                    centroids,
                    all_centroids,
                    hull,
                    vor_obj,
                    unified_regions,
                ) = analyze_voronoi_with_concave_hull(masks, img_bgr)

                detected = len(masks)
                success_rate = detected / n_particles * 100
                if rep == 0:
                    print(
                        f"    [F{vi}] Detection: {detected}/{n_particles} "
                        f"({success_rate:.1f}%)"
                    )

                variant_data = {
                    'name': f'Case_F_v{vi}_r{rep+1}',
                    'case': 'F',
                    'version': vi,
                    'repeat': rep + 1,
                    'image': img_bgr,
                    'positions_px': positions_px,
                    'sizes_nm': sizes,
                    'voronoi_areas': voronoi_areas,
                    'centroids': centroids,
                    'all_centroids': all_centroids,
                    'hull': hull,
                    'vor_obj': vor_obj,
                    'unified_regions': unified_regions,
                    'expected_di': '1.0',
                    'description': (
                        'Perfect uniform grid (no jitter), Variable size '
                        f'({min(sizes)}-{max(sizes)}nm), N={n_particles}'
                    ),
                }
                all_variants.append(variant_data)
                result = analyze_case(variant_data, save_visualization=True)
                if result:
                    all_results.append(result)

            print(f"    Version {vi}: {n_repeats} images completed")

    return all_variants, all_results


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_case(case_data, save_visualization=True):
    """Analyze single test case"""
    # Use precomputed Voronoi areas from case generation
    areas = case_data['voronoi_areas']
    
    if len(areas) == 0:
        print(f"  [FAIL] {case_data['name']}: No valid Voronoi areas")
        return None
    
    cv, ui = calculate_cv_di(areas)
    
    # Correlation analysis for variable-size uniform cases.
    correlation, p_value = np.nan, np.nan
    if case_data.get('case') in {'C', 'F'}:
        if len(areas) <= len(case_data['sizes_nm']):
            sizes_matched = case_data['sizes_nm'][:len(areas)]
            correlation, p_value = pearsonr(sizes_matched, areas)
    
    # Visualization (save for ALL repeats)
    if save_visualization:
        visualize_case(case_data, areas, cv, ui)
    
    return {
        'case_name': get_display_case_name(case_data) or case_data['name'],
        'case': case_data.get('case', case_data['name'].split('_')[1]),
        'version': case_data.get('version', 1),
        'repeat': case_data.get('repeat', 1),
        'description': case_data['description'],
        'num_particles': len(case_data['positions_px']),
        'num_voronoi_cells': len(areas),
        'cv': cv,
        'di': ui,
        'expected': case_data['expected_di'],
        'correlation': correlation,
        'p_value': p_value,
        'mean_area': np.mean(areas),
        'std_area': np.std(areas, ddof=1) if len(areas) > 1 else 0.0,
        'min_area': np.min(areas),
        'max_area': np.max(areas),
    }


def visualize_case(case_data, areas, cv, ui):
    """Create visualization for single case"""
    display_case_name = get_display_case_name(case_data) or case_data['name']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Synthetic image with concave hull
    ax1 = axes[0]
    ax1.imshow(cv2.cvtColor(case_data['image'], cv2.COLOR_BGR2RGB))
    
    # Draw concave hull if available
    if case_data['hull'] is not None:
        hull_np = np.array(case_data['hull'])
        ax1.plot(hull_np[:, 0], hull_np[:, 1], 'r-', linewidth=2, label='Concave Hull')
        ax1.legend()
    
    ax1.set_title(f"{display_case_name}\n{case_data['description']}", fontweight='bold')
    ax1.axis('off')
    
    # 2. Centroids with Voronoi overlay
    ax2 = axes[1]
    ax2.imshow(cv2.cvtColor(case_data['image'], cv2.COLOR_BGR2RGB), alpha=0.5)

    voronoi_plotted = False
    n_included = 0
    n_excluded = 0

    # Calculate excluded centroids (on hull boundary, not used for Voronoi)
    all_centroids = case_data.get('all_centroids', case_data['centroids'])
    included_centroids = case_data['centroids']

    if all_centroids is not None and included_centroids is not None:
        # Convert to sets for efficient exclusion calculation
        included_set = set(map(tuple, included_centroids))
        excluded_centroids = [c for c in all_centroids if tuple(c) not in included_set]

        # Plot EXCLUDED centroids first (background, green, lower zorder)
        if len(excluded_centroids) > 0:
            excluded_np = np.array(excluded_centroids)
            ax2.scatter(excluded_np[:, 0], excluded_np[:, 1],
                       s=60, c='limegreen', marker='x', linewidths=2,
                       label=f'Excluded ({len(excluded_centroids)})', zorder=4)
            n_excluded = len(excluded_centroids)

        # Plot INCLUDED centroids second (foreground, red, higher zorder)
        if len(included_centroids) > 0:
            centroids_np = np.array(included_centroids)
            ax2.scatter(centroids_np[:, 0], centroids_np[:, 1],
                       s=50, c='red', marker='o',
                       label=f'Included ({len(included_centroids)})', zorder=5)
            n_included = len(included_centroids)

        # Plot unified Voronoi regions (particle蹂??듯빀, concave hull 援먯감 ?곸슜)
        unified_regions = case_data.get('unified_regions', None)
        if unified_regions is not None:
            try:
                from shapely.geometry import Polygon, MultiPolygon
                # Draw unified polygon boundaries for each particle
                for key, polygon in unified_regions.items():
                    if isinstance(polygon, MultiPolygon):
                        for poly in polygon.geoms:
                            coords = np.array(poly.exterior.coords)
                            ax2.plot(coords[:, 0], coords[:, 1], 'b-', linewidth=1.5, alpha=0.6)
                    else:
                        coords = np.array(polygon.exterior.coords)
                        ax2.plot(coords[:, 0], coords[:, 1], 'b-', linewidth=1.5, alpha=0.6)
                voronoi_plotted = True
            except Exception as e:
                print(f"  [WARN] Voronoi plot failed: {e}")

    # Build title with particle counts
    total = n_included + n_excluded
    title = f"Voronoi Diagram\n"
    title += f"Total: {total} | Included: {n_included} | Excluded: {n_excluded}"

    if not voronoi_plotted and n_included > 0 and n_included < 4:
        title = f"Centroids Only (<4 for Voronoi)\n"
        title += f"Total: {total} | Included: {n_included} | Excluded: {n_excluded}"

    ax2.set_title(title, fontweight='bold', fontsize=10)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_xlim(0, case_data['image'].shape[1])
    ax2.set_ylim(case_data['image'].shape[0], 0)
    
    # 3. Area distribution with 100px짼 bin size
    ax3 = axes[2]

    # Calculate bins with 100px짼 intervals
    min_area = np.floor(np.min(areas) / 100) * 100
    max_area = np.ceil(np.max(areas) / 100) * 100
    bins = np.arange(min_area, max_area + 100, 100)
    
    ax3.hist(areas, bins=bins, color='steelblue', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Voronoi Cell Area (px짼)', fontweight='bold')
    ax3.set_ylabel('Frequency', fontweight='bold')
    ax3.set_title(f'Area Distribution\nCV={cv:.3f}, DI={ui:.3f}', fontweight='bold')
    
    # Y-axis: integer ticks only
    from matplotlib.ticker import MaxNLocator
    ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Display standard deviation in upper right corner (no Mean)
    std_area = np.std(areas, ddof=1) if len(areas) > 1 else 0.0
    ax3.text(0.95, 0.95, f'? = {std_area:.1f} px짼',
             transform=ax3.transAxes, fontsize=10, fontweight='bold',
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()

    # Save
    fig_path = os.path.join(OUTPUT_DIR, f'{display_case_name}_analysis.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    plt.close(fig)  # Explicitly close figure
    plt.clf()  # Clear figure


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def find_latest_final_excel(output_dir):
    """Find latest distribution_validation_FINAL_*.xlsx in output directory."""
    candidates = sorted(
        Path(output_dir).glob("distribution_validation_FINAL_*.xlsx"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _truthy_marker(value) -> bool:
    if pd.isna(value):
        return False
    s = str(value).strip().lower()
    return s in {"1", "true", "yes", "y", "t"}


def remap_case_name_with_mapping(value, mapping):
    """Replace the leading Case_X token with the supplied one-to-one mapping."""
    if pd.isna(value):
        return value

    text = str(value)

    def _replace(match):
        src = match.group(1).upper()
        dst = mapping.get(src, src)
        return f"Case_{dst}"

    return CASE_NAME_REMAP_PATTERN.sub(_replace, text, count=1)


def remap_case_name_value(value):
    """Keep case_name aligned with the legacy-to-internal case relabeling."""
    return remap_case_name_with_mapping(value, CASE_LABEL_REMAP)


def get_display_case_name(case_data):
    """Return the publication-order case label for titles and filenames."""
    if isinstance(case_data, dict) and "name" in case_data:
        return remap_case_name_value(case_data["name"])
    return None


def apply_case_label_remap(df: pd.DataFrame, marker_col: str = "case_label_remapped"):
    """
    Remap legacy case labels to alphabetical output labels once.

    Mapping:
      E->A, A->B, C->C, B->D, D->E, F->F
    """
    if "case" not in df.columns:
        return df, False

    out = df.copy()

    already_remapped = False
    if marker_col in out.columns:
        marker_vals = out[marker_col].dropna()
        if len(marker_vals) > 0:
            already_remapped = bool(marker_vals.map(_truthy_marker).all())

    out["case"] = out["case"].astype(str).str.upper().str.strip()

    case_name_changed = False

    if not already_remapped:
        out["case"] = out["case"].map(CASE_LABEL_REMAP).fillna(out["case"])
        remap_applied = True
    else:
        remap_applied = False

    if "case_name" in out.columns and not already_remapped:
        remapped_case_names = out["case_name"].map(remap_case_name_value)
        case_name_changed = not remapped_case_names.equals(out["case_name"])
        if case_name_changed:
            out["case_name"] = remapped_case_names

    remap_applied = bool(remap_applied or case_name_changed)

    out[marker_col] = True
    return out, remap_applied


def to_manuscript_case_labels(
    df: pd.DataFrame, *, include_scheme: bool = False
) -> pd.DataFrame:
    """Return an export copy whose A-F labels follow the manuscript order."""
    out = df.copy()
    if "case" not in out.columns:
        return out

    out["case"] = out["case"].astype(str).str.upper().str.strip()
    unknown = sorted(set(out["case"]) - set(MANUSCRIPT_CASE_REMAP))
    if unknown:
        raise ValueError(f"Unknown internal distribution cases: {unknown}")
    out["case"] = out["case"].map(MANUSCRIPT_CASE_REMAP)

    if "case_name" in out.columns:
        out["case_name"] = out["case_name"].map(
            lambda value: remap_case_name_with_mapping(
                value, MANUSCRIPT_CASE_REMAP
            )
        )

    if include_scheme:
        out["case_label_scheme"] = MANUSCRIPT_LABEL_SCHEME
    sort_columns = [
        column for column in ("case", "version", "repeat")
        if column in out.columns
    ]
    if sort_columns:
        out["case"] = pd.Categorical(
            out["case"], categories=MANUSCRIPT_CASE_ORDER, ordered=True
        )
        out = out.sort_values(sort_columns).reset_index(drop=True)
        out["case"] = out["case"].astype(str)
    return out


def from_manuscript_case_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Restore exported manuscript labels to the internal calculation labels."""
    out = df.copy()
    out["case"] = out["case"].astype(str).str.upper().str.strip()
    unknown = sorted(set(out["case"]) - set(MANUSCRIPT_CASE_REMAP_REVERSE))
    if unknown:
        raise ValueError(f"Unknown manuscript distribution cases: {unknown}")
    out["case"] = out["case"].map(MANUSCRIPT_CASE_REMAP_REVERSE)
    if "case_name" in out.columns:
        out["case_name"] = out["case_name"].map(
            lambda value: remap_case_name_with_mapping(
                value, MANUSCRIPT_CASE_REMAP_REVERSE
            )
        )
    out = out.drop(columns=["case_label_scheme"], errors="ignore")
    out["case_label_remapped"] = True
    return out


def rename_pair_columns_for_manuscript(df: pd.DataFrame) -> pd.DataFrame:
    """Translate pair identifiers embedded in summary column names."""
    rename_map = {}
    for column in df.columns:
        mapped_column = str(column)
        for left, right in PAIRWISE_DIFF_PAIRS:
            manuscript_left = MANUSCRIPT_CASE_REMAP[left]
            manuscript_right = MANUSCRIPT_CASE_REMAP[right]
            mapped_column = mapped_column.replace(
                f"{left}_lt_{right}",
                f"{manuscript_left}_lt_{manuscript_right}",
            )
            mapped_column = mapped_column.replace(
                f"delta_{left}{right}_",
                f"delta_{manuscript_left}{manuscript_right}_",
            )
        rename_map[column] = mapped_column
    return df.rename(columns=rename_map)


def load_results_df_from_final_excel(excel_path):
    """Load All_Results sheet from existing final excel and normalize schema."""
    with pd.ExcelFile(excel_path) as xls:
        target_sheet = (
            "All_Results" if "All_Results" in xls.sheet_names else xls.sheet_names[0]
        )
        df = pd.read_excel(xls, sheet_name=target_sheet)
    empty_unnamed = [
        column
        for column in df.columns
        if str(column).startswith('Unnamed:') and df[column].isna().all()
    ]
    if empty_unnamed:
        df = df.drop(columns=empty_unnamed)

    source_case_label_scheme = None
    if "case_label_scheme" in df.columns:
        schemes = sorted(
            set(
                df["case_label_scheme"]
                .dropna()
                .astype(str)
                .str.strip()
            )
        )
        if len(schemes) > 1:
            raise ValueError(f"Mixed case label schemes in final excel: {schemes}")
        if schemes:
            source_case_label_scheme = schemes[0]
            if source_case_label_scheme != MANUSCRIPT_LABEL_SCHEME:
                raise ValueError(
                    "Unsupported case label scheme in final excel: "
                    f"{source_case_label_scheme}"
                )

    # Normalize common column naming variants
    rename_map = {}
    for col in df.columns:
        c = str(col).strip()
        lc = c.lower()
        if lc in ("di", "ui"):
            rename_map[col] = "di"
        elif lc == "case":
            rename_map[col] = "case"
        elif lc == "version":
            rename_map[col] = "version"
        elif lc == "repeat":
            rename_map[col] = "repeat"
    if rename_map:
        df = df.rename(columns=rename_map)

    required = {"case", "version", "repeat", "di"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in final excel: {missing}")

    # Enforce numeric schema
    df["di"] = pd.to_numeric(df["di"], errors="coerce")
    df["version"] = pd.to_numeric(df["version"], errors="coerce")
    df["repeat"] = pd.to_numeric(df["repeat"], errors="coerce")
    df["case"] = df["case"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["case", "version", "repeat", "di"]).copy()
    df["version"] = df["version"].astype(int)
    df["repeat"] = df["repeat"].astype(int)
    if source_case_label_scheme == MANUSCRIPT_LABEL_SCHEME:
        df = from_manuscript_case_labels(df)
        remap_applied = False
        print(
            "[INFO] Restored manuscript A-F labels to internal labels for "
            "calculation."
        )
    else:
        df, remap_applied = apply_case_label_remap(df)
        if remap_applied:
            print(
                "[INFO] Applied legacy->alphabet case relabeling for reused "
                "final excel."
            )
        else:
            print("[INFO] Reused final excel already uses internal case labels.")
    df.attrs["legacy_case_remap_applied"] = bool(remap_applied)
    df.attrs["source_case_label_scheme"] = source_case_label_scheme

    return df


# Configuration
N_REPEATS = 10  # Number of repeats per Case/Version
MIN_GAP_NM = 5.0  # Minimum gap between particle edges (nm)
CACHE_ENABLED = True
FORCE_CACHE_REFRESH = os.environ.get("FORCE_CACHE_REFRESH", "0") == "1"
CACHE_ROOT_DIR = os.path.join(_project_root, "cache")

# Case selection
print("\n" + "=" * 80)
print("CASE SELECTION")
print("=" * 80)
print("Available Cases:")
print("  A - Same size + uniform distribution (with jitter)")
print("  B - Same size + clustered distribution")
print("  C - Variable size + uniform distribution (same positions as A)")
print("  D - Variable size + clustered distribution (same positions as B)")
print("  E - Perfect uniform distribution (no jitter)")
print("  F - Variable size + perfect uniform distribution (no jitter)")
print("  ALL - Run all cases (A, B, C, D, E, F)")
print("")

import argparse
parser = argparse.ArgumentParser(description="Distribution Validation")
parser.add_argument("--cases", type=str, default=None, help="Cases to run (e.g., A,B,D or ALL)")
parser.add_argument("--batch", action="store_true", help="Run in batch mode (auto-select ALL)")
parser.add_argument(
    "--final-excel",
    type=str,
    default=None,
    help="Use a specific distribution_validation_FINAL*.xlsx for plotting/statistics reuse",
)
parser.add_argument(
    "--force-rerun",
    action="store_true",
    help="Ignore existing FINAL excel and regenerate from scratch",
)
args, _ = parser.parse_known_args()

if args.cases:
    print(f"[CLI MODE] Using cases from argument: {args.cases}")
    user_input = args.cases.strip().upper()
elif args.batch or os.environ.get("BATCH_MODE") == "1" or not sys.stdin.isatty():
    print("[BATCH MODE] Auto-selecting ALL cases")
    user_input = "ALL"
else:
    print("Enter case(s) to run (e.g., 'A' or 'A,B,D' or 'ALL'):")
    try:
        user_input = input("> ").strip().upper()
    except EOFError:
        print("[BATCH MODE] EOF detected, auto-selecting ALL cases")
        user_input = "ALL"

if user_input in ("", "ALL"):
    selected_cases = ["A", "B", "C", "D", "E", "F"]
else:
    selected_cases = [c.strip() for c in user_input.split(",")]
    valid_cases = ["A", "B", "C", "D", "E", "F"]
    invalid = [c for c in selected_cases if c not in valid_cases]
    if invalid:
        print(f"[WARNING] Invalid case(s): {invalid}. Valid options: A, B, C, D, E, F, ALL")
        selected_cases = [c for c in selected_cases if c in valid_cases]
        if not selected_cases:
            print("[ERROR] No valid cases selected. Exiting.")
            sys.exit(1)

print(f"\n[OK] Selected cases: {', '.join(selected_cases)}")
print("[INFO] Output case relabeling enabled: E->A, A->B, C->C, B->D, D->E, F->F")

# Reuse existing FINAL excel (if available) to skip expensive rerun
loaded_from_final_excel = False
used_final_excel_path = None
preloaded_results_df = None
preloaded_case_remap_applied = False
preloaded_case_label_scheme = None
base_results_df = None
cases_to_generate = list(selected_cases)

if not args.force_rerun:
    final_excel_candidate = Path(args.final_excel).resolve() if args.final_excel else find_latest_final_excel(OUTPUT_DIR)
    if final_excel_candidate is not None and final_excel_candidate.exists():
        try:
            candidate_results_df = load_results_df_from_final_excel(str(final_excel_candidate))
            required_output_cases = {
                CASE_LABEL_REMAP.get(case, case) for case in selected_cases
            }
            available_output_cases = set(candidate_results_df['case'].astype(str))
            missing_output_cases = sorted(
                required_output_cases - available_output_cases
            )
            if missing_output_cases:
                base_results_df = candidate_results_df
                cases_to_generate = [
                    case
                    for case in selected_cases
                    if CASE_LABEL_REMAP.get(case, case) in missing_output_cases
                ]
                used_final_excel_path = str(final_excel_candidate)
                print("\n" + "=" * 80)
                print("APPENDING MISSING CASES TO EXISTING FINAL EXCEL")
                print("=" * 80)
                print(f"[OK] Base results: {used_final_excel_path}")
                print(f"[OK] Reusing {len(base_results_df)} existing result rows")
                print(
                    "[INFO] Only missing source cases will run through SAM: "
                    + ", ".join(cases_to_generate)
                )
            else:
                preloaded_results_df = candidate_results_df
                loaded_from_final_excel = True
                used_final_excel_path = str(final_excel_candidate)
                preloaded_case_remap_applied = bool(
                    preloaded_results_df.attrs.get("legacy_case_remap_applied", False)
                )
                preloaded_case_label_scheme = preloaded_results_df.attrs.get(
                    "source_case_label_scheme"
                )
                print("\n" + "=" * 80)
                print("REUSING EXISTING FINAL EXCEL")
                print("=" * 80)
                print(f"[OK] Loaded: {used_final_excel_path}")
                print(f"[OK] Rows: {len(preloaded_results_df)}")
                print("[INFO] Skipping synthetic generation / SAM inference.")
        except Exception as e:
            print(f"[WARN] Failed to reuse final excel ({type(e).__name__}: {e})")
            print("[INFO] Falling back to full generation pipeline.")

if preloaded_results_df is None:
    print("\n" + "=" * 80)
    print(f"GENERATING SYNTHETIC TEST CASES ({len(cases_to_generate)} Cases x 9 Versions x {N_REPEATS} Repeats)")
    print("=" * 80)
    print("Image Size: 256x256 pixels (512x512 nm)")
    print("Detection: SAM2 with 32x32 grid")
    print(f"Min Gap: {MIN_GAP_NM}nm between particles")
    print(f"Structure: {'/'.join(cases_to_generate)} x (23/25/27nm) x (25/36/49 particles)")
    print("=" * 80)

cache_key = f"cases_{'_'.join(cases_to_generate)}_repeats_{N_REPEATS}"
cache_options = {
    "selected_cases": cases_to_generate,
    "n_repeats": N_REPEATS,
    "min_gap_nm": MIN_GAP_NM,
    "generator_version": 3,
}

cases = []
all_results = []
total_elapsed = 0.0

if preloaded_results_df is not None:
    all_results = preloaded_results_df.to_dict(orient="records")
    # One row corresponds to one generated synthetic image result
    cases = [None] * len(all_results)
else:
    cached_run = None
    if CACHE_ENABLED and not FORCE_CACHE_REFRESH:
        cached_run = load_pipeline_cache(
            code_name=Path(__file__).stem,
            image_path=cache_key,
            options=cache_options,
            root_dir=CACHE_ROOT_DIR,
        )

    if isinstance(cached_run, dict) and "cases" in cached_run:
        cases = list(cached_run.get("cases", []))
        print(f"\n[OK] Cache hit: loaded {len(cases)} cached case entries")
        print("[INFO] Rebuilding visualizations from cached results...")
        for case_data in cases:
            result = analyze_case(case_data, save_visualization=True)
            if result:
                all_results.append(result)
        print(f"[OK] Rebuilt results from cache: {len(all_results)} rows")
    else:
        print("\n" + "=" * 80)
        print("LOADING SAM2 MODEL")
        print("=" * 80)
        sam_model, model_type = load_sam2_model()
        print(f"[OK] SAM2 {model_type} loaded on {device}")

        all_case_generators = {
            "A": ("CASE A", generate_case_a),
            "B": ("CASE B", generate_case_b),
            "C": ("CASE C", generate_case_c),
            "D": ("CASE D", generate_case_d),
            "E": ("CASE E", generate_case_e),
            "F": ("CASE F", generate_case_f),
        }
        case_generators = [
            (all_case_generators[c][0], all_case_generators[c][1])
            for c in cases_to_generate
        ]

        total_start = time.time()
        for case_idx, (case_name, generator_func) in enumerate(case_generators, start=1):
            print(f"\n[{case_idx}/{len(case_generators)}] Starting {case_name}...")
            case_start = time.time()
            batch, case_results = generator_func(sam_model, n_repeats=N_REPEATS)
            case_elapsed = time.time() - case_start
            cases.extend(batch)
            all_results.extend(case_results)
            print(
                f"[OK] [{case_idx}/{len(case_generators)}] {case_name} complete "
                f"({case_elapsed:.1f}s, {len(batch)} images, {len(case_results)} results)"
            )

        total_elapsed = time.time() - total_start
        print(f"\n[OK] ALL CASES GENERATED ({total_elapsed:.1f}s, {len(cases)} images, {len(all_results)} results)")

        if CACHE_ENABLED:
            save_pipeline_cache(
                code_name=Path(__file__).stem,
                image_path=cache_key,
                payload={"cases": cases, "all_results": all_results},
                options=cache_options,
                root_dir=CACHE_ROOT_DIR,
            )
            print("[OK] Saved distribution validation cache")

print("\n" + "=" * 80)
print("CREATING FINAL EXCEL FROM ACCUMULATED RESULTS")
print("=" * 80)

if len(all_results) > 0:
    generated_results_df = pd.DataFrame(all_results)
    generated_results_df, remap_applied = apply_case_label_remap(
        generated_results_df
    )
    if base_results_df is not None:
        results_df = pd.concat(
            [base_results_df, generated_results_df], ignore_index=True, sort=False
        )
        remap_applied = True
        print(
            f"  [OK] Merged {len(base_results_df)} reused rows with "
            f"{len(generated_results_df)} newly generated rows"
        )
    else:
        results_df = generated_results_df
    print(f"  [OK] Accumulated {len(results_df)} total results")
    if remap_applied:
        print(f"  [OK] Applied case relabeling: E->A, A->B, C->C, B->D, D->E, F->F")
else:
    print("  [WARN] No results accumulated")
    results_df = pd.DataFrame()

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def analyze_theoretical_order_violations(df):
    """
    Analyze same-distribution size-pair violations in the expected DI ordering.

    Returns DataFrame with violation statistics by version
    """
    print("\n" + "="*80)
    print("THEORETICAL ORDER VIOLATION ANALYSIS")
    print("="*80)

    violations = []

    versions = sorted(df['version'].unique())
    for version in versions:
        version_data = df[df['version'] == version]
        pivot = version_data.pivot_table(
            index='repeat', columns='case', values='di', aggfunc='mean'
        )
        row = {'version': version, 'n_repeats': int(len(pivot))}
        for left, right in PAIRWISE_DIFF_PAIRS:
            pair_key = f'{left}_lt_{right}'
            if left in pivot.columns and right in pivot.columns:
                paired = pivot[[left, right]].dropna()
                count = int((paired[left] < paired[right]).sum())
                pair_n = int(len(paired))
            else:
                count = 0
                pair_n = 0
            row[f'{pair_key}_N'] = pair_n
            row[f'{pair_key}_count'] = count
            row[f'{pair_key}_rate'] = count / pair_n if pair_n else np.nan
        violations.append(row)

    violations_df = pd.DataFrame(violations)

    if len(violations_df) > 0:
        total_row = {'version': 'TOTAL', 'n_repeats': violations_df['n_repeats'].sum()}
        for left, right in PAIRWISE_DIFF_PAIRS:
            pair_key = f'{left}_lt_{right}'
            pair_n = int(violations_df[f'{pair_key}_N'].sum())
            count = int(violations_df[f'{pair_key}_count'].sum())
            total_row[f'{pair_key}_N'] = pair_n
            total_row[f'{pair_key}_count'] = count
            total_row[f'{pair_key}_rate'] = count / pair_n if pair_n else np.nan
        violations_df = pd.concat([violations_df, pd.DataFrame([total_row])], ignore_index=True)

    print(f"\n  Versions analyzed: {len(versions)}")
    if len(violations_df) > 0:
        total = violations_df[violations_df['version'] == 'TOTAL'].iloc[0]
        for left, right in PAIRWISE_DIFF_PAIRS:
            pair_key = f'{left}_lt_{right}'
            rate = total[f'{pair_key}_rate']
            rate_text = f'{rate * 100:.1f}%' if np.isfinite(rate) else 'N/A'
            print(
                f"  {left} < {right} violations: "
                f"{int(total[f'{pair_key}_count'])}/"
                f"{int(total[f'{pair_key}_N'])} ({rate_text})"
            )

    return violations_df


def calculate_pairwise_comparisons(df):
    """
    Calculate paired DI differences for all configured size-control pairs.

    Returns DataFrame with delta DI statistics by version
    """
    print("\n" + "="*80)
    print("PAIRWISE DICOMPARISON ANALYSIS")
    print("="*80)

    comparisons = []

    for version in sorted(df['version'].unique()):
        version_data = df[df['version'] == version]
        pivot = version_data.pivot_table(
            index='repeat', columns='case', values='di', aggfunc='mean'
        )
        comparison_row = {'version': version, 'n_repeats': int(len(pivot))}
        for left, right in PAIRWISE_DIFF_PAIRS:
            key = f'{left}{right}'
            if left not in pivot.columns or right not in pivot.columns:
                continue
            deltas = (pivot[left] - pivot[right]).dropna().to_numpy(dtype=float)
            if len(deltas) == 0:
                continue
            comparison_row.update(
                {
                    f'delta_{key}_N': int(len(deltas)),
                    f'delta_{key}_mean': float(np.mean(deltas)),
                    f'delta_{key}_std': float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                    f'delta_{key}_min': float(np.min(deltas)),
                    f'delta_{key}_max': float(np.max(deltas)),
                }
            )
        comparisons.append(comparison_row)

    comparisons_df = pd.DataFrame(comparisons)

    print(f"\n  Analyzed {len(comparisons_df)} versions")
    for left, right in PAIRWISE_DIFF_PAIRS:
        key = f'{left}{right}'
        mean_column = f'delta_{key}_mean'
        std_column = f'delta_{key}_std'
        if mean_column in comparisons_df.columns:
            print(
                f"  {left}-{right} mean: "
                f"{comparisons_df[mean_column].mean():.4f} +/- "
                f"{comparisons_df[std_column].mean():.4f}"
            )

    return comparisons_df


def calculate_version_statistics(df):
    """
    Calculate detailed statistics for each Case-Version combination

    Returns DataFrame with per-version statistics
    """
    print("\n" + "="*80)
    print("VERSION-LEVEL STATISTICS")
    print("="*80)

    version_stats = df.groupby(['case', 'version']).agg({
        'di': ['mean', 'std', 'min', 'max'],
        'cv': ['mean', 'std', 'min', 'max'],
        'num_particles': 'mean',
        'num_voronoi_cells': 'mean',
        'mean_area': 'mean',
        'std_area': 'mean',
    })

    # Flatten column names
    version_stats.columns = ['_'.join(col).strip('_') for col in version_stats.columns]
    version_stats = version_stats.reset_index()

    print(f"\n  Generated statistics for {len(version_stats)} case-version combinations")

    return version_stats


def bootstrap_mean_ci(values, n_resamples=10000, ci=95, seed=42):
    """Bootstrap mean and percentile CI."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    boot_means = arr[idx].mean(axis=1)
    alpha = (100 - ci) / 2
    return float(arr.mean()), float(np.percentile(boot_means, alpha)), float(np.percentile(boot_means, 100 - alpha))


def format_p_value(p):
    """Format p-value for compact figure annotations."""
    if p is None or not np.isfinite(p):
        return "N/A"
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.4f}"


def format_metric(value, fmt=".3f"):
    """Format scalar metric with finite-value guard."""
    if value is None or not np.isfinite(value):
        return "N/A"
    return format(value, fmt)


def build_pairwise_diff_long(df):
    """
    Build long-format paired differences DataFrame:
      columns = [pair, version, repeat, delta_DI]
      pairs = A-F, B-C, D-E
    """
    required_cols = {"case", "version", "repeat", "DI"}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns for pairwise diff: {missing_cols}")

    pivot = df.pivot_table(
        index=["version", "repeat"],
        columns="case",
        values="DI",
        aggfunc="mean",
    )

    records = []
    for a, b in PAIRWISE_DIFF_PAIRS:
        if a not in pivot.columns or b not in pivot.columns:
            continue
        sub = pivot[[a, b]].dropna()
        deltas = sub[a] - sub[b]
        for (version, repeat), delta in deltas.items():
            records.append(
                {
                    "pair": f"{a}-{b}",
                    "version": int(version),
                    "repeat": int(repeat),
                    "delta_DI": float(delta),
                }
            )

    return pd.DataFrame(records)


def build_manuscript_stat_tables(df):
    """Build raw SUI values and checks referenced by the manuscript."""
    from scipy.stats import wilcoxon

    required = {"case", "version", "repeat", "di"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing SUI manuscript columns: {missing}")

    source = df.copy()
    source["case"] = source["case"].astype(str).str.upper().str.strip()
    for column in ("version", "repeat", "di"):
        source[column] = pd.to_numeric(source[column], errors="coerce")
    finite_rows = np.isfinite(source[["version", "repeat", "di"]]).all(axis=1)
    source = source.loc[finite_rows].copy()
    source["version"] = source["version"].astype(int)
    source["repeat"] = source["repeat"].astype(int)

    duplicate_keys = source.duplicated(["case", "version", "repeat"]).any()
    expected_keys = {
        (case, version, repeat)
        for case in THEORETICAL_CASE_ORDER
        for version in range(1, 10)
        for repeat in range(1, N_REPEATS + 1)
    }
    observed_keys = set(
        source[["case", "version", "repeat"]].itertuples(
            index=False, name=None
        )
    )
    publication_ready = bool(
        not duplicate_keys
        and observed_keys == expected_keys
        and len(source) == len(expected_keys)
    )

    condition_rows = []
    for index, code_case in enumerate(THEORETICAL_CASE_ORDER):
        values = source.loc[source["case"] == code_case, "di"].to_numpy(
            dtype=float
        )
        mean_sui, ci_low, ci_high = bootstrap_mean_ci(
            values, n_resamples=10000, ci=95, seed=42 + index
        )
        q1, median, q3 = (
            np.quantile(values, [0.25, 0.50, 0.75])
            if len(values)
            else (np.nan, np.nan, np.nan)
        )
        condition_rows.append(
            {
                "Code_Case": code_case,
                "Manuscript_Case": MANUSCRIPT_CASE_REMAP[code_case],
                "Condition": CONDITION_DEFINITIONS[code_case],
                "Theoretical_Order_Index": index + 1,
                "N": int(len(values)),
                "Mean_SUI": mean_sui,
                "SUI_CI95_Low": ci_low,
                "SUI_CI95_High": ci_high,
                "SD_SUI_ddof1": (
                    float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
                ),
                "Median_SUI": float(median),
                "SUI_Q1": float(q1),
                "SUI_Q3": float(q3),
                "SUI_IQR": float(q3 - q1),
            }
        )
    condition_summary = pd.DataFrame(condition_rows)

    diff_long = build_pairwise_diff_long(
        source[["case", "version", "repeat", "di"]].rename(
            columns={"di": "DI"}
        )
    )
    pair_rows = []
    for index, (left, right) in enumerate(PAIRWISE_DIFF_PAIRS):
        code_pair = f"{left}-{right}"
        manuscript_pair = (
            f"{MANUSCRIPT_CASE_REMAP[left]}-"
            f"{MANUSCRIPT_CASE_REMAP[right]}"
        )
        values = diff_long.loc[
            diff_long["pair"] == code_pair, "delta_DI"
        ].to_numpy(dtype=float)
        mean_delta, ci_low, ci_high = bootstrap_mean_ci(
            values, n_resamples=10000, ci=95, seed=142 + index
        )
        try:
            _, wilcoxon_p = wilcoxon(values, alternative="greater")
        except ValueError:
            wilcoxon_p = np.nan
        delta_sd = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
        pair_rows.append(
            {
                "Code_Pair": code_pair,
                "Manuscript_Pair": manuscript_pair,
                "Delta_Direction": (
                    f"{MANUSCRIPT_CASE_REMAP[left]} minus "
                    f"{MANUSCRIPT_CASE_REMAP[right]}"
                ),
                "N": int(len(values)),
                "Mean_Delta_SUI": mean_delta,
                "Delta_SUI_CI95_Low": ci_low,
                "Delta_SUI_CI95_High": ci_high,
                "Delta_SUI_SD_ddof1": delta_sd,
                "Cohen_dz": (
                    float(mean_delta / delta_sd)
                    if np.isfinite(delta_sd) and delta_sd > 0.0
                    else np.nan
                ),
                "Wilcoxon_OneSided_Greater_P": float(wilcoxon_p),
                "Violation_Count_Left_Less_Than_Right": int(
                    np.sum(values < 0.0)
                ),
                "Violation_Percent_Left_Less_Than_Right": (
                    float(np.mean(values < 0.0) * 100.0)
                    if len(values)
                    else np.nan
                ),
            }
        )
    paired_summary = pd.DataFrame(pair_rows)

    pivot = source.pivot_table(
        index=["version", "repeat"], columns="case", values="di",
        aggfunc="mean",
    )
    positional_rows = []
    for index, (left, right) in enumerate(POSITIONAL_ORDER_PAIRS):
        paired = pivot[[left, right]].dropna()
        deltas = (paired[left] - paired[right]).to_numpy(dtype=float)
        mean_delta, ci_low, ci_high = bootstrap_mean_ci(
            deltas, n_resamples=10000, ci=95, seed=242 + index
        )
        violation_mask = deltas <= 0.0
        positional_rows.append(
            {
                "Code_Comparison": f"{left}>{right}",
                "Manuscript_Comparison": (
                    f"{MANUSCRIPT_CASE_REMAP[left]}>"
                    f"{MANUSCRIPT_CASE_REMAP[right]}"
                ),
                "N": int(len(deltas)),
                "Mean_Delta_SUI": mean_delta,
                "Delta_SUI_CI95_Low": ci_low,
                "Delta_SUI_CI95_High": ci_high,
                "Violation_Count_Left_Less_Than_Or_Equal_Right": int(
                    violation_mask.sum()
                ),
                "Violation_Percent_Left_Less_Than_Or_Equal_Right": (
                    float(violation_mask.mean() * 100.0)
                    if len(deltas)
                    else np.nan
                ),
            }
        )
    positional_summary = pd.DataFrame(positional_rows)

    condition_definitions = condition_summary[
        [
            "Code_Case",
            "Manuscript_Case",
            "Condition",
            "Theoretical_Order_Index",
        ]
    ].copy()

    def manuscript_row(item, value, unit, definition):
        return {
            "Item": item,
            "Value": value,
            "Unit": unit,
            "Definition": definition,
        }

    manuscript_rows = [
        manuscript_row(
            "Publication_Ready", publication_ready, "boolean",
            "Exactly six cases x nine versions x ten repeats with unique keys",
        ),
        manuscript_row(
            "Synthetic_Result_Count", len(source), "rows",
            "All finite case/version/repeat SUI observations",
        ),
        manuscript_row(
            "Case_Count", source["case"].nunique(), "cases",
            "Synthetic conditions represented in All_Results",
        ),
        manuscript_row(
            "Version_Count", source["version"].nunique(), "versions",
            "Synthetic particle-count/size versions",
        ),
        manuscript_row(
            "Repeat_Count_Per_Case_Version", N_REPEATS, "repeats",
            "Configured repeats for every case-version combination",
        ),
        manuscript_row(
            "SUI_Formula", "1 / (1 + CV_A)", "formula",
            "CV_A is the coefficient of variation of Voronoi cell areas",
        ),
        manuscript_row(
            "Voronoi_Area_SD_ddof", 1, "integer",
            "Sample standard deviation uses denominator n-1",
        ),
        manuscript_row(
            "Bootstrap_Resamples", 10000, "resamples",
            "Percentile bootstrap used for mean SUI and paired deltas",
        ),
        manuscript_row(
            "Confidence_Level", 0.95, "proportion",
            "Two-sided percentile bootstrap confidence interval",
        ),
    ]
    for _, row_data in condition_summary.iterrows():
        prefix = f"Case_{row_data['Manuscript_Case']}"
        for field, unit in (
            ("N", "observations"),
            ("Mean_SUI", "SUI"),
            ("SUI_CI95_Low", "SUI"),
            ("SUI_CI95_High", "SUI"),
        ):
            manuscript_rows.append(
                manuscript_row(
                    f"{prefix}_{field}", row_data[field], unit,
                    f"Manuscript case {row_data['Manuscript_Case']}",
                )
            )
    for _, row_data in paired_summary.iterrows():
        prefix = f"Pair_{row_data['Manuscript_Pair'].replace('-', '_')}"
        for field, unit in (
            ("N", "paired observations"),
            ("Mean_Delta_SUI", "SUI"),
            ("Delta_SUI_CI95_Low", "SUI"),
            ("Delta_SUI_CI95_High", "SUI"),
            ("Cohen_dz", "standardized effect"),
            ("Wilcoxon_OneSided_Greater_P", "p value"),
        ):
            manuscript_rows.append(
                manuscript_row(
                    f"{prefix}_{field}", row_data[field], unit,
                    (
                        f"Manuscript pair {row_data['Manuscript_Pair']}; "
                        "left minus right"
                    ),
                )
            )
    manuscript_values = pd.DataFrame(
        manuscript_rows, columns=["Item", "Value", "Unit", "Definition"]
    )

    condition_definitions = condition_definitions.drop(
        columns=["Code_Case"]
    ).rename(columns={"Manuscript_Case": "Case"})
    condition_summary = condition_summary.drop(
        columns=["Code_Case"]
    ).rename(columns={"Manuscript_Case": "Case"})
    paired_summary = paired_summary.drop(
        columns=["Code_Pair"]
    ).rename(columns={"Manuscript_Pair": "Pair"})
    positional_summary = positional_summary.drop(
        columns=["Code_Comparison"]
    ).rename(columns={"Manuscript_Comparison": "Comparison"})

    return {
        "condition_definitions": condition_definitions,
        "condition_summary": condition_summary,
        "paired_summary": paired_summary,
        "positional_summary": positional_summary,
        "manuscript_values": manuscript_values,
        "publication_ready": publication_ready,
    }


def create_publication_figure(df, df_diff, output_dir, timestamp):
    """
    Create publication-quality 1x2 multi-panel figures.
    - main_figure: with theory/stat annotation text
    - main_figure_notext: without theory/stat annotation text
    """
    import seaborn as sns
    from scipy.stats import wilcoxon

    _ = timestamp  # kept for backwards-compatible function signature

    cases_order = MANUSCRIPT_CASE_ORDER
    pairs_order = list(MANUSCRIPT_PAIR_REMAP.values())

    # User-specified custom palette
    custom_palette = ["#BF5065", "#4B7BA6", "#58A65D", "#D96D55"]
    internal_case_colors = {
        "A": custom_palette[1],  # legacy E
        "B": custom_palette[2],  # legacy A
        "C": custom_palette[3],  # legacy C
        "D": custom_palette[0],  # legacy B
        "E": custom_palette[1],  # legacy D
        "F": custom_palette[0],  # new perfect-uniform variable-size control
    }
    case_colors = {
        MANUSCRIPT_CASE_REMAP[case]: color
        for case, color in internal_case_colors.items()
    }
    internal_pair_colors = {
        "A-F": custom_palette[0],
        "B-C": custom_palette[1],
        "D-E": custom_palette[3],
    }
    pair_colors = {
        MANUSCRIPT_PAIR_REMAP[pair]: color
        for pair, color in internal_pair_colors.items()
    }

    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    plot_df = df.copy()
    plot_df["case"] = plot_df["case"].map(MANUSCRIPT_CASE_REMAP)
    plot_df = plot_df[plot_df["case"].isin(cases_order)].copy()
    diff_plot = df_diff.copy()
    diff_plot["pair"] = diff_plot["pair"].map(MANUSCRIPT_PAIR_REMAP)
    diff_plot = diff_plot[diff_plot["pair"].isin(pairs_order)].copy()

    def _compute_dz(vals: np.ndarray) -> float:
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size < 2:
            return np.nan
        std = float(np.std(vals, ddof=1))
        if std <= 0:
            return np.nan
        return float(np.mean(vals) / std)

    def _draw_and_save(show_theory_text: bool, show_stats_text: bool, base_name: str):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.6))

        # (a) DI distributions by case
        sns.violinplot(
            data=plot_df,
            x="case",
            y="DI",
            order=cases_order,
            palette=case_colors,
            inner=None,
            cut=0,
            linewidth=0.8,
            ax=ax_a,
        )
        sns.boxplot(
            data=plot_df,
            x="case",
            y="DI",
            order=cases_order,
            width=0.18,
            showfliers=False,
            boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 0.8, "alpha": 0.9},
            whiskerprops={"linewidth": 0.8, "color": "black"},
            capprops={"linewidth": 0.8, "color": "black"},
            medianprops={"linewidth": 1.0, "color": "black"},
            ax=ax_a,
        )
        sns.stripplot(
            data=plot_df,
            x="case",
            y="DI",
            order=cases_order,
            color="black",
            alpha=0.45,
            size=2.2,
            jitter=0.18,
            ax=ax_a,
        )

        means_by_case = (
            plot_df.groupby("case", as_index=True)["DI"]
            .mean()
            .reindex(cases_order)
        )
        valid_mask = means_by_case.notna().to_numpy()
        if valid_mask.any():
            x_pos = np.arange(len(cases_order))[valid_mask]
            y_pos = means_by_case.to_numpy(dtype=float)[valid_mask]
            ax_a.scatter(x_pos, y_pos, s=40, c="black", marker="o", zorder=7)

        ax_a.set_ylim(0.5, 1.02)
        ax_a.set_xlabel("Case")
        ax_a.set_ylabel("DI")
        ax_a.set_title("DI distributions by case")
        ax_a.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.2)
        ax_a.text(
            -0.18,
            1.04,
            "(a)",
            transform=ax_a.transAxes,
            fontsize=10,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

        if show_theory_text:
            info_text = (
                f"Theoretical order: {MANUSCRIPT_THEORETICAL_ORDER_TEXT}\n"
                "DI = 1/(1+CV), CV = σ/μ (Voronoi cell area)"
            )
            ax_a.text(
                0.02,
                0.03,
                info_text,
                transform=ax_a.transAxes,
                fontsize=7.4,
                ha="left",
                va="bottom",
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor="0.6",
                    alpha=0.92,
                ),
            )

        # (b) Paired differences (delta_DI)
        sns.violinplot(
            data=diff_plot,
            x="pair",
            y="delta_DI",
            order=pairs_order,
            palette=pair_colors,
            inner=None,
            cut=0,
            linewidth=0.8,
            ax=ax_b,
        )
        sns.stripplot(
            data=diff_plot,
            x="pair",
            y="delta_DI",
            order=pairs_order,
            color="black",
            alpha=0.45,
            size=2.2,
            jitter=0.18,
            ax=ax_b,
        )

        ax_b.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
        ax_b.set_ylim(-0.01, 0.02)
        ax_b.set_xlabel("Pair")
        ax_b.set_ylabel("ΔDI")
        ax_b.set_title("Paired ΔDI distributions")
        ax_b.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.2)
        ax_b.text(
            -0.18,
            1.04,
            "(b)",
            transform=ax_b.transAxes,
            fontsize=10,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

        if show_stats_text:
            y_top = ax_b.get_ylim()[1]
            y_span = ax_b.get_ylim()[1] - ax_b.get_ylim()[0]
            y_text = y_top - 0.03 * y_span

            for i, pair in enumerate(pairs_order):
                vals = diff_plot.loc[
                    diff_plot["pair"] == pair, "delta_DI"
                ].dropna().to_numpy(dtype=float)

                if vals.size == 0:
                    ann = "mean=N/A\n95% CI [N/A, N/A]\nWilcoxon p=N/A\ndz=N/A"
                else:
                    mean_delta, ci_low, ci_high = bootstrap_mean_ci(
                        vals, n_resamples=10000, ci=95, seed=42 + i
                    )
                    try:
                        _, p_val = wilcoxon(vals, alternative="greater")
                    except ValueError:
                        p_val = np.nan
                    dz = _compute_dz(vals)

                    ann = (
                        f"mean={format_metric(mean_delta, '.3f')}\n"
                        f"95% CI [{format_metric(ci_low, '.3f')}, {format_metric(ci_high, '.3f')}]\n"
                        f"Wilcoxon p={format_p_value(p_val)}\n"
                        f"Cohen's dz={format_metric(dz, '.3f')}"
                    )

                ax_b.text(
                    i,
                    y_text,
                    ann,
                    ha="center",
                    va="top",
                    fontsize=7.1,
                    bbox=dict(
                        boxstyle="round,pad=0.22",
                        facecolor="white",
                        edgecolor="0.6",
                        alpha=0.92,
                    ),
                )

        sns.despine(fig=fig)
        plt.tight_layout(w_pad=1.6)

        strip_all_text = (not show_theory_text) and (not show_stats_text)
        if strip_all_text:
            for fig_text in fig.texts[:]:
                fig_text.set_visible(False)
            for ax in fig.get_axes():
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()
                ax.set_title("")
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.tick_params(
                    axis="x",
                    labelbottom=False,
                    labeltop=False,
                    bottom=True,
                    top=False,
                )
                ax.tick_params(
                    axis="y",
                    labelleft=False,
                    labelright=False,
                    left=True,
                    right=False,
                )
                for txt in ax.texts[:]:
                    txt.remove()

        pdf_path = os.path.join(output_dir, f"{base_name}.pdf")
        png_path = os.path.join(output_dir, f"{base_name}.png")
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        return pdf_path, png_path

    main_pdf, main_png = _draw_and_save(
        show_theory_text=True,
        show_stats_text=True,
        base_name="main_figure",
    )
    notext_pdf, notext_png = _draw_and_save(
        show_theory_text=False,
        show_stats_text=False,
        base_name="main_figure_notext",
    )

    return {
        "pdf": main_pdf,
        "png": main_png,
        "notext_pdf": notext_pdf,
        "notext_png": notext_png,
    }


# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING FINAL EXCEL")
print("="*80)

if len(results_df) > 0:
    # Calculate averages by CASE only (not version)
    print("  Computing averages by Case...")
    case_avg = results_df.groupby(['case']).agg({
        'num_particles': 'mean',
        'num_voronoi_cells': 'mean',
        'cv': ['mean', 'std'],
        'di': ['mean', 'std'],
        'correlation': 'mean',
        'p_value': 'mean',
        'mean_area': 'mean',
        'std_area': 'mean',
    })

    # Flatten column names
    case_avg.columns = ['_'.join(col).strip('_') for col in case_avg.columns]
    case_avg = case_avg.reset_index()

    # Add expected DI(take first occurrence for each case)
    expected_by_case = results_df.groupby(['case']).first()[['expected']].reset_index()
    case_avg = case_avg.merge(expected_by_case, on=['case'])

    # Reorder columns
    case_avg = case_avg[['case', 'expected',
                         'num_particles_mean', 'num_voronoi_cells_mean',
                         'di_mean', 'di_std', 'cv_mean', 'cv_std',
                         'correlation_mean', 'p_value_mean', 'mean_area_mean']]
    case_avg['case'] = pd.Categorical(case_avg['case'], categories=CASE_ORDER_ALPHABET, ordered=True)
    case_avg = case_avg.sort_values('case').reset_index(drop=True)
    case_avg['case'] = case_avg['case'].astype(str)

    # ========== NEW: Advanced Statistical Analysis ==========
    print("\n  Running advanced statistical analysis...")

    # Version-level statistics
    version_stats_df = calculate_version_statistics(results_df)

    # Theoretical order violation analysis
    violations_df = analyze_theoretical_order_violations(results_df)

    # Pairwise comparisons (delta DI analysis)
    pairwise_df = calculate_pairwise_comparisons(results_df)

    # Raw manuscript values, condition CIs, footprint-pair effects, and
    # positional-order exception counts.
    manuscript_tables = build_manuscript_stat_tables(results_df)

    # Keep internal labels for calculations, but expose only the A-F notation
    # used in the manuscript in every saved result sheet.
    results_export_df = to_manuscript_case_labels(
        results_df, include_scheme=True
    ).drop(columns=["case_label_remapped"], errors="ignore")
    case_avg_export_df = to_manuscript_case_labels(case_avg)
    version_stats_export_df = to_manuscript_case_labels(version_stats_df)
    violations_export_df = rename_pair_columns_for_manuscript(violations_df)
    pairwise_export_df = rename_pair_columns_for_manuscript(pairwise_df)

    # Excel output (reuse existing FINAL excel if loaded)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    required_manuscript_sheets = {
        'Condition_Definitions',
        'Condition_Summary_CI',
        'Paired_Bootstrap_Summary',
        'Positional_Order_Summary',
        'Manuscript Values',
    }
    missing_manuscript_sheets = set(required_manuscript_sheets)
    if loaded_from_final_excel and used_final_excel_path:
        with pd.ExcelFile(used_final_excel_path) as existing_excel:
            missing_manuscript_sheets -= set(existing_excel.sheet_names)
    should_rewrite_excel = bool(
        (not loaded_from_final_excel)
        or preloaded_case_remap_applied
        or preloaded_case_label_scheme != MANUSCRIPT_LABEL_SCHEME
        or missing_manuscript_sheets
    )
    if (not should_rewrite_excel) and loaded_from_final_excel and used_final_excel_path:
        excel_path = used_final_excel_path
        print(f"\n[INFO] Reusing existing FINAL excel: {excel_path}")
        print(f"  [INFO] Skip rewriting excel; using loaded results for plotting/statistics.")
    else:
        if loaded_from_final_excel and preloaded_case_remap_applied:
            print("\n[INFO] Legacy case labels were remapped; writing updated FINAL excel.")
        if (
            loaded_from_final_excel
            and preloaded_case_label_scheme != MANUSCRIPT_LABEL_SCHEME
        ):
            print(
                "\n[INFO] Rewriting FINAL excel with manuscript A-F labels."
            )
        if loaded_from_final_excel and used_final_excel_path:
            excel_path = used_final_excel_path
            if missing_manuscript_sheets:
                print(
                    "\n[INFO] Adding manuscript statistic sheets to the existing "
                    f"FINAL excel: {sorted(missing_manuscript_sheets)}"
                )
        else:
            excel_path = os.path.join(
                OUTPUT_DIR, f"distribution_validation_FINAL_{timestamp}.xlsx"
            )

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Sheet 1: All individual results (with case and version)
            results_export_df.to_excel(writer, sheet_name='All_Results', index=False)

            # Sheet 2: Averages by case (A-F)
            case_avg_export_df.to_excel(writer, sheet_name='Case_Averages', index=False)

            # Sheet 3: Version-level statistics
            version_stats_export_df.to_excel(writer, sheet_name='Version_Statistics', index=False)

            # Sheet 4: Theoretical order violations
            violations_export_df.to_excel(writer, sheet_name='Order_Violations', index=False)

            # Sheet 5: Pairwise comparisons (delta DI)
            pairwise_export_df.to_excel(writer, sheet_name='Pairwise_DeltaUI', index=False)

            manuscript_tables['condition_definitions'].to_excel(
                writer, sheet_name='Condition_Definitions', index=False
            )
            manuscript_tables['condition_summary'].to_excel(
                writer, sheet_name='Condition_Summary_CI', index=False
            )
            manuscript_tables['paired_summary'].to_excel(
                writer, sheet_name='Paired_Bootstrap_Summary', index=False
            )
            manuscript_tables['positional_summary'].to_excel(
                writer, sheet_name='Positional_Order_Summary', index=False
            )
            manuscript_tables['manuscript_values'].to_excel(
                writer, sheet_name='Manuscript Values', index=False
            )

        print(f"\n??Final Excel saved: {excel_path}")
        print(f"  ?뱤 Sheet 1: All_Results ({len(results_export_df)} rows)")
        print(f"  ?뱤 Sheet 2: Case_Averages ({len(case_avg_export_df)} rows)")
        print(f"  ?뱤 Sheet 3: Version_Statistics ({len(version_stats_export_df)} rows)")
        print(f"  ?뱤 Sheet 4: Order_Violations ({len(violations_export_df)} rows)")
        print(f"  ?뱤 Sheet 5: Pairwise_DeltaDI({len(pairwise_export_df)} rows)")
        print(
            "  Manuscript sheets: "
            f"conditions={len(manuscript_tables['condition_summary'])}, "
            f"footprint_pairs={len(manuscript_tables['paired_summary'])}, "
            f"positional_checks={len(manuscript_tables['positional_summary'])}, "
            f"ready={manuscript_tables['publication_ready']}"
        )
else:
    print("  ?좑툘 No results to save")
    case_avg = pd.DataFrame()
    excel_path = None

# ============================================================================
# VISUALIZATION: PUBLICATION FIGURE (1x2)
# ============================================================================

if len(results_df) > 0:
    print("\n" + "=" * 80)
    print("GENERATING PUBLICATION FIGURE (1x2)")
    print("=" * 80)

    try:
        # Build long-format df expected by publication figure generator
        df_plot = results_df[['case', 'version', 'repeat', 'di']].copy()
        df_plot = df_plot.rename(columns={'di': 'DI'})
        df_plot['version'] = pd.to_numeric(df_plot['version'], errors='coerce').astype('Int64')
        df_plot['repeat'] = pd.to_numeric(df_plot['repeat'], errors='coerce').astype('Int64')
        df_plot = df_plot.dropna(subset=['case', 'version', 'repeat', 'DI']).copy()
        df_plot['version'] = df_plot['version'].astype(int)
        df_plot['repeat'] = df_plot['repeat'].astype(int)

        # Build paired-difference DataFrame for configured size-control pairs.
        df_diff = build_pairwise_diff_long(df_plot)

        fig_paths = create_publication_figure(df_plot, df_diff, OUTPUT_DIR, timestamp)
        publication_pdf_path = fig_paths['pdf']
        publication_png_path = fig_paths['png']
        publication_notext_pdf_path = fig_paths.get('notext_pdf')
        publication_notext_png_path = fig_paths.get('notext_png')

        print(f"  [OK] Publication PDF saved: {publication_pdf_path}")
        print(f"  [OK] Publication PNG saved: {publication_png_path}")
        if publication_notext_pdf_path:
            print(f"  [OK] Publication NoText PDF saved: {publication_notext_pdf_path}")
        if publication_notext_png_path:
            print(f"  [OK] Publication NoText PNG saved: {publication_notext_png_path}")

    except Exception as e:
        print(f"  [FAIL] Publication figure generation failed: {e}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("VALIDATION SUMMARY (Case Averages)")
print("="*80)

summary_case_avg = (
    case_avg_export_df if 'case_avg_export_df' in locals() else case_avg
)
if len(summary_case_avg) > 0:
    print(f"\n{'Case':<6} {'DI_mean':<10} {'DI_std':<10} {'CV_mean':<10} {'CV_std':<10} {'Expected':<15}")
    print("-" * 70)
    for _, row in summary_case_avg.iterrows():
        print(f"{row['case']:<6} {row['di_mean']:<10.4f} {row['di_std']:<10.4f} {row['cv_mean']:<10.4f} {row['cv_std']:<10.4f} {row['expected']:<15}")

# Print violation summary if available
if len(results_df) > 0 and 'violations_df' in locals():
    print("\n" + "="*80)
    print("THEORETICAL ORDER VIOLATIONS SUMMARY")
    print("="*80)

    total_violations = violations_df[violations_df['version'] == 'TOTAL']
    if len(total_violations) > 0:
        total_row = total_violations.iloc[0]
        print(f"\nTotal Comparisons: {int(total_row['n_repeats'])}")
        for left, right in PAIRWISE_DIFF_PAIRS:
            pair_key = f'{left}_lt_{right}'
            rate = total_row[f'{pair_key}_rate']
            rate_text = f'{rate * 100:.1f}%' if np.isfinite(rate) else 'N/A'
            manuscript_left = MANUSCRIPT_CASE_REMAP[left]
            manuscript_right = MANUSCRIPT_CASE_REMAP[right]
            print(
                f"  {manuscript_left} < {manuscript_right} violations: "
                f"{int(total_row[f'{pair_key}_count'])}/"
                f"{int(total_row[f'{pair_key}_N'])} ({rate_text})"
            )
        print(f"\nInterpretation:")
        print(f"  - Lower violation rates indicate better conformity to theoretical order")
        for left, right in PAIRWISE_DIFF_PAIRS:
            print(
                "  - Expected size-control relation: "
                f"{MANUSCRIPT_CASE_REMAP[left]} >= "
                f"{MANUSCRIPT_CASE_REMAP[right]}"
            )

print("\n" + "="*80)
print("VALIDATION COMPLETE")
print("="*80)
print(f"\nTotal images generated: {len(cases)}")
print(f"Total results analyzed: {len(results_df) if len(results_df) > 0 else 0}")
print(f"Output directory: {OUTPUT_DIR}")
if excel_path:
    print(f"  ?뱤 Final Excel: {excel_path}")
    if 'publication_pdf_path' in locals():
        print(f"  [FIG] Publication PDF: {publication_pdf_path}")
    if 'publication_png_path' in locals():
        print(f"  [FIG] Publication PNG (600 dpi): {publication_png_path}")
    if 'publication_notext_pdf_path' in locals():
        print(f"  [FIG] Publication NoText PDF: {publication_notext_pdf_path}")
    if 'publication_notext_png_path' in locals():
        print(f"  [FIG] Publication NoText PNG (600 dpi): {publication_notext_png_path}")
print(f"Total runtime: {total_elapsed:.1f}s")

print("\n" + "="*80)
print("APPLIED FEATURES")
print("="*80)
print("?뱤 DETECTION:")
print("  ??SAM2 automatic mask generation")
print("  ??Overlap removal (IoU > 0.5)")
print("  ??Background filtering (4-border OR >90% area)")
print("  ??IQR outlier filtering")
print("\n?뱪 DISTRIBUTION ANALYSIS:")
print("  ??Concave hull (alpha shape) boundary detection")
print("  ??Unified Voronoi areas (boundary pixel consolidation)")
print("  ??Hull-intersected Voronoi cells")
print(f"\n?뱢 STATISTICS:")
print(f"  ??{N_REPEATS} repeats per Case/Version for statistical significance")
print(f"  ??Mean and Std calculated for DI and CV")

