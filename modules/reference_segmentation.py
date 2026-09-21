"""Reference-mask loading and object-level segmentation evaluation."""

from __future__ import annotations

import base64
import json
import zlib
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


REFERENCE_EVALUATION_VERSION = "datasetninja_crop_hungarian_assignment_v1"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def find_reference_annotation_path(image_stem: str, annotation_dir) -> Optional[Path]:
    """Find a DatasetNinja annotation JSON for an image stem."""
    annotation_dir = Path(annotation_dir)
    if not annotation_dir.is_dir():
        return None
    candidates = sorted(annotation_dir.glob(f"{image_stem}.*.json"))
    if candidates:
        return candidates[0]
    direct_path = annotation_dir / f"{image_stem}.json"
    return direct_path if direct_path.is_file() else None


def find_reference_image_path(image_stem: str, image_dir) -> Optional[Path]:
    """Find the full-size source image corresponding to an image stem."""
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        return None
    candidates = sorted(
        path
        for path in image_dir.glob(f"{image_stem}.*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    return candidates[0] if candidates else None


def decode_datasetninja_bitmap(bitmap_data: str) -> np.ndarray:
    """Decode a DatasetNinja bitmap payload into a boolean array."""
    if "base64," in bitmap_data:
        bitmap_data = bitmap_data.split("base64,", 1)[1]
    compact_data = "".join(bitmap_data.split())
    decoded = base64.b64decode(compact_data)
    if len(decoded) > 2 and decoded[0] == 0x78:
        decoded = zlib.decompress(decoded)
    mask = cv2.imdecode(np.frombuffer(decoded, np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("Failed to decode DatasetNinja bitmap")
    return mask > 0


def find_crop_offset(original_image: np.ndarray, cropped_image: np.ndarray):
    """Locate an exact or near-exact dataset crop in its full-size source image."""
    original_h, original_w = original_image.shape[:2]
    crop_h, crop_w = cropped_image.shape[:2]
    if crop_h > original_h or crop_w > original_w:
        raise ValueError(
            f"Dataset crop {cropped_image.shape[:2]} exceeds source image "
            f"{original_image.shape[:2]}"
        )
    result = cv2.matchTemplate(original_image, cropped_image, cv2.TM_CCOEFF_NORMED)
    _, max_value, _, max_location = cv2.minMaxLoc(result)
    return int(max_location[0]), int(max_location[1]), float(max_value)


def load_datasetninja_reference_masks(
    cropped_image_path,
    annotation_dir,
    reference_image_dir,
    target_class: Optional[str] = "particle",
):
    """Load full-image annotations and align them to a cropped Dataset image."""
    cropped_image_path = Path(cropped_image_path)
    image_stem = cropped_image_path.stem
    annotation_path = find_reference_annotation_path(image_stem, annotation_dir)
    reference_image_path = find_reference_image_path(image_stem, reference_image_dir)
    if annotation_path is None:
        raise FileNotFoundError(f"Reference annotation not found for {image_stem}")
    if reference_image_path is None:
        raise FileNotFoundError(f"Reference source image not found for {image_stem}")

    cropped_image = cv2.imread(str(cropped_image_path), cv2.IMREAD_GRAYSCALE)
    reference_image = cv2.imread(str(reference_image_path), cv2.IMREAD_GRAYSCALE)
    if cropped_image is None:
        raise ValueError(f"Failed to read cropped image: {cropped_image_path}")
    if reference_image is None:
        raise ValueError(f"Failed to read reference image: {reference_image_path}")

    offset_x, offset_y, match_confidence = find_crop_offset(
        reference_image,
        cropped_image,
    )
    crop_h, crop_w = cropped_image.shape[:2]
    annotation_data = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotation_h = int(
        annotation_data.get("size", {}).get("height", reference_image.shape[0])
    )
    annotation_w = int(
        annotation_data.get("size", {}).get("width", reference_image.shape[1])
    )

    masks = []
    object_ids = []
    skipped_non_target = 0
    for object_index, annotation_object in enumerate(annotation_data.get("objects", [])):
        class_title = annotation_object.get("classTitle")
        if target_class and class_title != target_class:
            skipped_non_target += 1
            continue
        bitmap = annotation_object.get("bitmap", {})
        if "data" not in bitmap or "origin" not in bitmap:
            continue

        bitmap_mask = decode_datasetninja_bitmap(bitmap["data"])
        origin_x, origin_y = (int(value) for value in bitmap["origin"])
        bitmap_h, bitmap_w = bitmap_mask.shape[:2]
        source_x1 = max(origin_x, 0)
        source_y1 = max(origin_y, 0)
        source_x2 = min(origin_x + bitmap_w, annotation_w)
        source_y2 = min(origin_y + bitmap_h, annotation_h)
        if source_x2 <= source_x1 or source_y2 <= source_y1:
            continue

        full_mask = np.zeros((annotation_h, annotation_w), dtype=bool)
        bitmap_x1 = source_x1 - origin_x
        bitmap_y1 = source_y1 - origin_y
        bitmap_x2 = bitmap_x1 + (source_x2 - source_x1)
        bitmap_y2 = bitmap_y1 + (source_y2 - source_y1)
        full_mask[source_y1:source_y2, source_x1:source_x2] = bitmap_mask[
            bitmap_y1:bitmap_y2,
            bitmap_x1:bitmap_x2,
        ]

        crop_x2 = min(offset_x + crop_w, annotation_w)
        crop_y2 = min(offset_y + crop_h, annotation_h)
        cropped_mask = np.zeros((crop_h, crop_w), dtype=bool)
        cropped_region = full_mask[offset_y:crop_y2, offset_x:crop_x2]
        cropped_mask[: cropped_region.shape[0], : cropped_region.shape[1]] = cropped_region
        if not np.any(cropped_mask):
            continue
        masks.append(cropped_mask)
        object_ids.append(str(annotation_object.get("id", object_index)))

    annotation_stat = annotation_path.stat()
    reference_image_stat = reference_image_path.stat()
    return {
        "gt_masks": masks,
        "object_ids": object_ids,
        "crop_offset": (offset_x, offset_y),
        "crop_match_confidence": match_confidence,
        "crop_shape": (crop_h, crop_w),
        "annotation_path": str(annotation_path.resolve()),
        "annotation_size_bytes": int(annotation_stat.st_size),
        "annotation_mtime_ns": int(annotation_stat.st_mtime_ns),
        "reference_image_path": str(reference_image_path.resolve()),
        "reference_image_size_bytes": int(reference_image_stat.st_size),
        "reference_image_mtime_ns": int(reference_image_stat.st_mtime_ns),
        "target_class": target_class,
        "skipped_non_target_objects": skipped_non_target,
        "evaluation_version": REFERENCE_EVALUATION_VERSION,
    }


def crop_reference_masks(gt_masks: Iterable[np.ndarray], crop_box, border_margin: int = 5):
    """Crop reference masks to an analysis region and flag truncated objects."""
    x1, y1, x2, y2 = (int(value) for value in crop_box)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid reference crop box: {crop_box}")
    region_h = y2 - y1
    region_w = x2 - x1
    cropped_masks = []
    boundary_flags = []
    for mask in gt_masks:
        mask = np.asarray(mask, dtype=bool)
        region = np.ascontiguousarray(mask[y1:y2, x1:x2])
        if region.shape != (region_h, region_w):
            padded = np.zeros((region_h, region_w), dtype=bool)
            padded[: region.shape[0], : region.shape[1]] = region
            region = padded
        if not np.any(region):
            continue
        ys, xs = np.where(region)
        boundary_flags.append(
            bool(
                xs.min() <= border_margin
                or ys.min() <= border_margin
                or xs.max() >= region_w - 1 - border_margin
                or ys.max() >= region_h - 1 - border_margin
            )
        )
        cropped_masks.append(region)
    return cropped_masks, boundary_flags


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Calculate exact pixel IoU for two binary masks."""
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    if mask_a.shape != mask_b.shape:
        raise ValueError(f"Mask shape mismatch: {mask_a.shape} != {mask_b.shape}")
    intersection = np.logical_and(mask_a, mask_b).sum(dtype=np.int64)
    union = np.logical_or(mask_a, mask_b).sum(dtype=np.int64)
    return float(intersection / union) if union else 0.0


def hungarian_assignment_scores(gt_masks, predicted_masks):
    """Return threshold-independent Hungarian assignments and their IoU scores."""
    gt_masks = [np.asarray(mask, dtype=bool) for mask in gt_masks]
    predicted_masks = [
        np.asarray(
            mask.get("segmentation") if isinstance(mask, dict) else mask,
            dtype=bool,
        )
        for mask in predicted_masks
    ]
    iou_matrix = np.zeros((len(gt_masks), len(predicted_masks)), dtype=np.float64)
    if not gt_masks or not predicted_masks:
        return [], iou_matrix
    for gt_index, gt_mask in enumerate(gt_masks):
        for pred_index, predicted_mask in enumerate(predicted_masks):
            iou_matrix[gt_index, pred_index] = mask_iou(gt_mask, predicted_mask)
    row_indices, column_indices = linear_sum_assignment(-iou_matrix)
    assignments = [
        {
            "gt_index": int(gt_index),
            "pred_index": int(pred_index),
            "iou": float(iou_matrix[gt_index, pred_index]),
        }
        for gt_index, pred_index in zip(row_indices, column_indices)
    ]
    return assignments, iou_matrix


def metrics_from_assignment_scores(n_gt: int, n_pred: int, assignments, tau: float):
    """Threshold cached assignment scores into TP, FP, FN and derived metrics."""
    tau = float(tau)
    if not 0.0 <= tau <= 1.0:
        raise ValueError(f"tau must be between 0 and 1, got {tau}")
    matched_assignments = [
        assignment for assignment in assignments if float(assignment["iou"]) >= tau
    ]
    true_positive = len(matched_assignments)
    false_positive = int(n_pred) - true_positive
    false_negative = int(n_gt) - true_positive
    precision = true_positive / n_pred if n_pred else 0.0
    recall = true_positive / n_gt if n_gt else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    matched_ious = [float(assignment["iou"]) for assignment in matched_assignments]
    return {
        "tau": tau,
        "n_gt": int(n_gt),
        "n_pred": int(n_pred),
        "true_positive": int(true_positive),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "matched_ious": matched_ious,
    }
