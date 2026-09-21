"""
SAM2 Utilities
==============
Helper functions for SAM2 model loading with correct paths.
"""

import os
from pathlib import Path

def get_sam2_config_path(config_name):
    """
    Get absolute path to SAM2 config file from installed package.

    Args:
        config_name: Config filename (e.g., 'sam2.1_hiera_l.yaml')

    Returns:
        str: Absolute path to config file

    Example:
        >>> config = get_sam2_config_path('sam2.1_hiera_l.yaml')
        >>> sam = build_sam2(config, checkpoint_path, device)
    """
    try:
        import sam2
        sam2_package_path = os.path.dirname(sam2.__file__)
        config_path = os.path.join(sam2_package_path, 'configs', 'sam2.1', config_name)

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                f"Available configs should be in: {os.path.dirname(config_path)}"
            )

        return config_path

    except ImportError:
        raise ImportError(
            "SAM2 is not installed. Install it with:\n"
            "pip install git+https://github.com/facebookresearch/sam2.git"
        )


def get_sam2_checkpoint_map():
    """
    Get checkpoint configuration map for SAM2 models.

    Returns:
        dict: Mapping of model types to (checkpoint_filename, config_name)

    Note:
        Checkpoint files should be in 'checkpoints/' directory.
        Config files are automatically found from SAM2 package.
    """
    # Use absolute path based on this module's location
    module_dir = Path(__file__).parent.parent  # Entire_Framework directory
    checkpoints_dir = module_dir / "checkpoints"
    
    checkpoint_map = {
        "hiera_l": (str(checkpoints_dir / "sam2.1_hiera_large.pt"), "sam2.1_hiera_l.yaml"),
        "hiera_b+": (str(checkpoints_dir / "sam2.1_hiera_base_plus.pt"), "sam2.1_hiera_b+.yaml"),
        "hiera_s": (str(checkpoints_dir / "sam2.1_hiera_small.pt"), "sam2.1_hiera_s.yaml"),
        "hiera_t": (str(checkpoints_dir / "sam2.1_hiera_tiny.pt"), "sam2.1_hiera_t.yaml")
    }
    return checkpoint_map


def load_sam2_model(model_type="hiera_l", device=None):
    """
    Load SAM2 model with automatic config path resolution.

    Args:
        model_type: Model size ('hiera_l', 'hiera_b+', 'hiera_s', 'hiera_t')
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        SAM2 model instance

    Example:
        >>> sam = load_sam2_model('hiera_l', 'cuda')
    """
    from sam2.build_sam import build_sam2

    import torch

    checkpoint_map = get_sam2_checkpoint_map()
    model_type = "hiera_l"
    checkpoint_path, config_name = checkpoint_map[model_type]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get absolute config path from SAM2 package
    config_path = get_sam2_config_path(config_name)

    # Check checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Download from: https://github.com/facebookresearch/sam2#download-checkpoints"
        )

    print(f"Loading SAM2 model: {model_type}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Config: {config_path}")
    print(f"  Device: {device}")

    sam = build_sam2(config_path, checkpoint_path, device=device)

    print("SAM2 model loaded successfully")
    return sam


# Alias for backward compatibility
load_sam_model = load_sam2_model


SAM_POSTPROCESSING_VERSION = (
    "background_bbox90_border5_then_largest_contour_centroid_containment_v1"
)


def filter_background_masks(
    masks: list,
    image_shape,
    border_tolerance: int = 5,
    max_bbox_coverage: float = 0.90,
) -> list:
    """Remove full-image background masks using the shared experiment rule."""
    height, width = image_shape[:2]
    total_pixels = max(height * width, 1)
    filtered = []

    for mask_dict in masks:
        x, y, box_width, box_height = mask_dict["bbox"]
        coverage = (box_width * box_height) / total_pixels
        if coverage > max_bbox_coverage:
            continue

        touches_all_borders = (
            x <= border_tolerance
            and y <= border_tolerance
            and (x + box_width) >= (width - border_tolerance)
            and (y + box_height) >= (height - border_tolerance)
        )
        if touches_all_borders:
            continue
        filtered.append(mask_dict)

    return filtered


def get_largest_contour_centroid(mask):
    """Return the largest external contour centroid used by all experiments."""
    import cv2
    import numpy as np

    contours, _ = cv2.findContours(
        np.asarray(mask, dtype=np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(largest_contour)
    if moments["m00"] <= 0:
        return None
    return (
        int(moments["m10"] / moments["m00"]),
        int(moments["m01"] / moments["m00"]),
    )


def filter_overlapping_masks_by_centroid(masks: list) -> list:
    """Remove the larger of centroid-contained duplicate SAM masks."""
    import numpy as np

    if not masks:
        return masks

    centroids = [
        get_largest_contour_centroid(mask_dict["segmentation"])
        for mask_dict in masks
    ]
    areas = [
        int(np.count_nonzero(mask_dict["segmentation"]))
        for mask_dict in masks
    ]
    masks_to_keep = set(range(len(masks)))

    for i, mask_dict in enumerate(masks):
        if i not in masks_to_keep or centroids[i] is None:
            continue
        mask = np.asarray(mask_dict["segmentation"], dtype=bool)

        for j, centroid in enumerate(centroids):
            if j == i or j not in masks_to_keep or centroid is None:
                continue
            x, y = centroid
            if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
                continue
            if not mask[y, x]:
                continue

            # The existing comparison/validation rule removes the larger mask.
            # Equal-area duplicates keep the earlier mask deterministically.
            if areas[i] > areas[j]:
                masks_to_keep.discard(i)
                break
            masks_to_keep.discard(j)

    return [masks[index] for index in sorted(masks_to_keep)]


def postprocess_sam_masks(
    masks: list,
    image_shape,
    border_tolerance: int = 5,
    max_bbox_coverage: float = 0.90,
) -> list:
    """Apply the canonical post-processing shared by experiments and GUI."""
    background_filtered = filter_background_masks(
        masks,
        image_shape,
        border_tolerance=border_tolerance,
        max_bbox_coverage=max_bbox_coverage,
    )
    return filter_overlapping_masks_by_centroid(background_filtered)


def generate_masks(model, image: "np.ndarray",
                   points_per_side: int = 32,
                   points_per_batch: int = 256,
                   pred_iou_thresh: float = 0.95,
                   stability_score_thresh: float = 0.80,
                   crop_n_layers: int = 1,
                   crop_n_points_downscale_factor: int = 2,
                   crop_nms_thresh: float = 0.7,
                   box_nms_thresh: float = 0.7,
                   use_m2m: bool = True,
                   filter_background: bool = True) -> list:
    """
    Generate masks using SAM2 AutomaticMaskGenerator.

    UNIFIED SAM2 PARAMETERS (consistent across all scripts)

    Args:
        model: Loaded SAM2 model
        image: Input image (grayscale or RGB)
        points_per_side: Number of points per side for grid sampling
        points_per_batch: Batch size for point processing
        pred_iou_thresh: IoU threshold for mask filtering
        stability_score_thresh: Stability threshold for mask filtering
        crop_n_layers: Number of crop layers for multi-scale detection
        crop_n_points_downscale_factor: Downscale factor for crop points
        crop_nms_thresh: NMS threshold for crop masks
        box_nms_thresh: NMS threshold for bounding boxes
        use_m2m: Use mask-to-mask refinement
        filter_background: Apply the legacy background-only filter before return

    Returns:
        List of mask dicts (each with 'segmentation', 'bbox', 'area', etc.)
    """
    import numpy as np
    import cv2
    import torch
    from torchvision.ops.boxes import batched_nms, box_area
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.utils.amg import MaskData, generate_crop_boxes

    class EmptyMaskSafeSAM2AutomaticMaskGenerator(SAM2AutomaticMaskGenerator):
        """Preserve valid zero-mask results in sam-2==1.0 multi-crop runs."""

        def _process_crop(self, crop_image, crop_box, crop_layer_idx, orig_size):
            data = super()._process_crop(
                crop_image, crop_box, crop_layer_idx, orig_size
            )
            if len(data["rles"]) == 0:
                data["boxes"] = data["boxes"].reshape(0, 4)
                data["points"] = data["points"].reshape(0, 2)
                data["crop_boxes"] = data["crop_boxes"].reshape(0, 4)
            return data

        def _generate_masks(self, crop_image):
            orig_size = crop_image.shape[:2]
            crop_boxes, layer_idxs = generate_crop_boxes(
                orig_size, self.crop_n_layers, self.crop_overlap_ratio
            )
            data = MaskData()
            for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
                crop_data = self._process_crop(
                    crop_image, crop_box, layer_idx, orig_size
                )
                data.cat(crop_data)

            if len(crop_boxes) > 1 and len(data["rles"]) > 0:
                scores = (1 / box_area(data["crop_boxes"])).to(
                    data["boxes"].device
                )
                keep_by_nms = batched_nms(
                    data["boxes"].float(),
                    scores,
                    torch.zeros_like(data["boxes"][:, 0]),
                    iou_threshold=self.crop_nms_thresh,
                )
                data.filter(keep_by_nms)
            data.to_numpy()
            return data

    # Convert grayscale to RGB if needed
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        # VISION's image-upload and preprocessing pipeline uses OpenCV BGR.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create mask generator with unified parameters for TEM nanoparticles
    mask_generator = EmptyMaskSafeSAM2AutomaticMaskGenerator(
        model,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=crop_n_layers,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        crop_nms_thresh=crop_nms_thresh,
        box_nms_thresh=box_nms_thresh,
        use_m2m=use_m2m
    )

    print(f"Generating masks with SAM2:")
    print(f"  - points_per_side={points_per_side}")
    print(f"  - pred_iou_thresh={pred_iou_thresh}")
    print(f"  - stability_score_thresh={stability_score_thresh}")
    print(f"  - crop_n_layers={crop_n_layers}")
    print(f"  - use_m2m={use_m2m}")

    # Generate masks
    masks_data = mask_generator.generate(image_rgb)

    if not filter_background:
        print(f"Generated {len(masks_data)} raw masks")
        return masks_data

    filtered_masks = filter_background_masks(masks_data, image.shape[:2])
    print(f"Generated {len(masks_data)} masks, background-filtered to {len(filtered_masks)}")
    return filtered_masks
