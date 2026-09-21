"""
Shape Validation Pipeline
- Sample images from dataset
- Run SAM2 + CLIP pipeline with full preprocessing
- GUI annotation tool for ground truth labeling
- Incremental annotation (reuse existing annotations)
- Calculate validation metrics (accuracy, confusion matrix, etc.)
- Generate visualizations and Excel reports
"""
import os
import sys
import argparse
import hashlib
import pickle
import re
# Add parent directory to path for modules access
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)


import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.gridspec import GridSpec
import seaborn as sns
from collections import Counter
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
    balanced_accuracy_score,
)
import torch
from PIL import Image

# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# CLIP imports
import clip

# Module imports (from existing codebase)
from modules.preprocessing import preprocess_with_config
# MAD noise estimation removed - using fixed sigma=40 for BM3D
import base64
import zlib
from modules.runtime_config import preprocessing_record, preprocessing_record_matches
from modules.project_paths import get_validation_dir, get_optimization_dir
from modules.sam2_utils import (
    filter_background_masks as filter_sam_background_masks,
    filter_overlapping_masks_by_centroid,
)


class ShapeValidator:
    """
    Shape validation pipeline for SAM2 + CLIP framework

    Features:
    - Full preprocessing pipeline (BM3D + Noise2SR)
    - SAM2 segmentation with background removal
    - CLIP shape classification with crop & zoom
    - Interactive GUI annotation tool
    - Incremental annotation (reuse existing files)
    - Comprehensive metrics and visualizations
    """

    # Preset configurations for 2D and 3D shapes
    SHAPE_PRESETS = {
        '2D': {
            'labels': ['Circle', 'Triangle', 'Quadrilateral', 'Hexagon', 'Irregular'],
            'descriptions': {
                'Circle': 'circle',
                'Triangle': ['equilateral triangle', 'isosceles triangle', 'scalene triangle'],
                'Quadrilateral': ['square', 'rhombus', 'rectangle', 'rhomboid', 'isosceles trapezium', 'trapezium'],
                'Hexagon': 'hexagon',
                'Irregular': 'irregular',
            }
        },

        '3D': {
            'labels': ['Spheroid', 'Pyramid', 'Hexahedron', 'Cylinder', 'Irregular'],
            'descriptions': {
                'Spheroid': ['oblate spheroid', 'prolate spheroid', 'sphere'],
                'Pyramid': 'tetrahedron',
                'Hexahedron': ['cube', 'parallelpiped'],
                'Cylinder': ['disc', 'tube'],
                'Irregular': 'irregular',
            }
        }
    }

    def __init__(self, dataset_dir: str, output_dir: str, shape_labels: Optional[List[str]] = None,
                 shape_preset: str = '3D', pred_iou_thresh: float = 0.95,
                 stability_score_thresh: float = 0.80,
                 expected_images: Optional[int] = None):
        """
        Initialize ShapeValidator

        Args:
            dataset_dir: Path to cropped images folder
            output_dir: Path to output folder
            shape_labels: List of shape category names (default: use preset)
            shape_preset: '2D' or '3D' preset configuration (default: '3D')
        """
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir) / "shape_validation"
        self.pred_iou_thresh = float(pred_iou_thresh)
        self.stability_score_thresh = float(stability_score_thresh)
        self.expected_images = (
            int(expected_images) if expected_images is not None else None
        )
        if not 0.0 <= self.pred_iou_thresh <= 1.0:
            raise ValueError("pred_iou_thresh must be between 0 and 1")
        if not 0.0 <= self.stability_score_thresh <= 1.0:
            raise ValueError("stability_score_thresh must be between 0 and 1")
        self.sam_config = {
            'points_per_side': 32,
            'points_per_batch': 256,
            'pred_iou_thresh': self.pred_iou_thresh,
            'stability_score_thresh': self.stability_score_thresh,
            'crop_n_layers': 1,
            'crop_n_points_downscale_factor': 2,
            'crop_nms_thresh': 0.7,
            'box_nms_thresh': 0.7,
            'use_m2m': True,
        }

        # Shape labels (use preset or custom)
        if shape_labels is None:
            if shape_preset in self.SHAPE_PRESETS:
                self.shape_labels = self.SHAPE_PRESETS[shape_preset]['labels']
                self.shape_preset = shape_preset
            else:
                raise ValueError(f"Invalid preset: {shape_preset}. Use '2D' or '3D'")
        else:
            self.shape_labels = shape_labels
            self.shape_preset = None

        # Generate CLIP descriptions
        self.shape_descriptions = self._generate_descriptions()
        self.clip_prompts = [
            (
                f"This is a {label} nanoparticle on electron microscope image. "
                f"Shape description: {description}"
            )
            for label, description in zip(self.shape_labels, self.shape_descriptions)
        ]

        # Create output directories
        self.annotations_dir = self.output_dir / "annotations"
        self.predictions_dir = self.output_dir / "predictions"
        self.results_dir = self.output_dir / "results"
        self.viz_dir = self.results_dir / "visualizations"
        self.preprocessed_dir = self.output_dir / "preprocessed_bm3d_noise2sr"
        self.compare_preprocess_dir = get_optimization_dir("preprocessing_comparison_full")

        for dir_path in [
            self.annotations_dir,
            self.predictions_dir,
            self.results_dir,
            self.viz_dir,
            self.preprocessed_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Models (lazy loading)
        self._predictor = None  # SAM2 predictor
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_text_tokens = None

        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Fixed color map for all shape labels (consistent across all images)
        np.random.seed(42)  # Fixed seed for reproducibility
        self.shape_color_map = {}
        for label in self.shape_labels:
            # Generate random but distinct colors
            color_rgb = tuple(np.random.randint(50, 200, 3).tolist())  # Avoid too dark/bright
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])  # RGB to BGR for OpenCV
            self.shape_color_map[label] = {
                'rgb': color_rgb,
                'bgr': color_bgr,
                'hex': '#{:02x}{:02x}{:02x}'.format(*color_rgb)
            }

        print(f"ShapeValidator initialized")
        print(f"  Shape categories: {len(self.shape_labels)}")
        print(f"  Device: {self.device}")
        print(f"  Output: {self.output_dir}")
        print(f"  BM3D+Noise2SR cache: {self.preprocessed_dir}")
        print(
            "  SAM2 thresholds: "
            f"pred_iou={self.pred_iou_thresh:.2f}, "
            f"stability={self.stability_score_thresh:.2f}"
        )

    def _generate_descriptions(self) -> List[str]:
        """
        Generate CLIP text descriptions for each shape label

        Returns:
            List of text descriptions for CLIP
        """
        descriptions = []

        for label in self.shape_labels:
            # Check if using preset configuration
            if self.shape_preset and label in self.SHAPE_PRESETS[self.shape_preset]['descriptions']:
                desc_info = self.SHAPE_PRESETS[self.shape_preset]['descriptions'][label]

                # If multiple descriptions (list), use all of them
                if isinstance(desc_info, list):
                    desc_text = ', '.join(desc_info)
                else:
                    desc_text = desc_info

                descriptions.append(desc_text)
            else:
                # Custom label - use simple description
                descriptions.append(label)

        return descriptions

    def _sam_config_matches(self, cached_config: object) -> bool:
        """Return True only when recorded SAM settings match this validator run."""
        if not isinstance(cached_config, dict):
            return False
        for key, expected in self.sam_config.items():
            actual = cached_config.get(key)
            if isinstance(expected, bool):
                matches = isinstance(actual, bool) and actual == expected
            elif isinstance(expected, (int, float)):
                try:
                    matches = bool(np.isclose(
                        float(actual), float(expected), rtol=0.0, atol=1e-12
                    ))
                except (TypeError, ValueError):
                    matches = False
            else:
                matches = actual == expected
            if not matches:
                return False
        return True

    # =========================================================================
    # MASK ENCODING/DECODING UTILITIES
    # =========================================================================

    @staticmethod
    def encode_mask(mask: np.ndarray) -> str:
        """
        Encode boolean mask to compressed base64 string

        Args:
            mask: Boolean numpy array (H x W)

        Returns:
            Compressed base64 encoded string
        """
        # Convert to uint8 (0 or 255)
        mask_uint8 = (mask.astype(np.uint8) * 255)
        # Compress with zlib
        compressed = zlib.compress(mask_uint8.tobytes(), level=9)
        # Encode to base64
        encoded = base64.b64encode(compressed).decode('utf-8')
        return encoded

    @staticmethod
    def decode_mask(encoded: str, shape: Tuple[int, int]) -> np.ndarray:
        """
        Decode compressed base64 string to boolean mask

        Args:
            encoded: Compressed base64 encoded string
            shape: Original mask shape (H, W)

        Returns:
            Boolean numpy array
        """
        # Decode from base64
        compressed = base64.b64decode(encoded.encode('utf-8'))
        # Decompress
        mask_bytes = zlib.decompress(compressed)
        # Convert back to numpy array
        mask_uint8 = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(shape)
        # Convert to boolean
        mask_bool = (mask_uint8 > 0)
        return mask_bool

    # =========================================================================
    # BORDER DETECTION
    # =========================================================================

    @staticmethod
    def is_border_touching(mask: np.ndarray, border_tolerance: int = 2) -> bool:
        """
        Check if a particle mask touches any image border

        Args:
            mask: Boolean or uint8 mask array (H x W)
            border_tolerance: Pixels from edge to consider as border

        Returns:
            True if mask touches any border
        """
        # Ensure mask is boolean
        if mask.dtype != bool:
            mask = mask > 0

        h, w = mask.shape

        # Check each border
        touches_top = np.any(mask[:border_tolerance, :])
        touches_bottom = np.any(mask[h-border_tolerance:, :])
        touches_left = np.any(mask[:, :border_tolerance])
        touches_right = np.any(mask[:, w-border_tolerance:])

        return touches_top or touches_bottom or touches_left or touches_right

    # =========================================================================
    # MODEL LOADING
    # =========================================================================

    def load_sam2_model(self):
        """Load SAM2 model (high quality settings)"""
        if self._predictor is not None:
            return self._predictor

        print("\nLoading SAM2 model...")

        sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        sam2 = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
        # UNIFIED SAM2 PARAMETERS (consistent across all validation scripts)
        self._predictor = SAM2AutomaticMaskGenerator(
            sam2,
            points_per_side=32,                    # Standard grid density
            points_per_batch=256,                  # Batch size for point processing
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            crop_n_layers=1,                       # Multi-scale cropping
            crop_n_points_downscale_factor=2,      # Point density in crops
            crop_nms_thresh=0.7,                   # NMS threshold for crops
            box_nms_thresh=0.7,                    # NMS threshold for boxes
            use_m2m=True                           # Mask-to-mask refinement
        )

        print(f"[SUCCESS] SAM2 loaded on {self.device}")
        return self._predictor

    def load_clip_model(self):
        """Load CLIP model"""
        if self._clip_model is not None:
            return self._clip_model, self._clip_preprocess, self._clip_text_tokens

        print("\nLoading CLIP model...")

        # Load CLIP (ViT-L/14@336px - highest accuracy model)
        self._clip_model, self._clip_preprocess = clip.load("ViT-L/14@336px", device=self.device)
        self._clip_model.eval()

        # Tokenize text descriptions
        self._clip_text_tokens = clip.tokenize(self.clip_prompts).to(self.device)

        print(f"[SUCCESS] CLIP loaded on {self.device}")
        print(f"          Categories: {', '.join(self.shape_labels)}")
        for label, prompt in zip(self.shape_labels, self.clip_prompts):
            print(f"          {label}: {prompt}")

        return self._clip_model, self._clip_preprocess, self._clip_text_tokens

    # =========================================================================
    # PREPROCESSING (BM3D + Noise2SR cache)
    # =========================================================================

    def _bm3d_noise2sr_cache_path(self, image_name: str) -> Path:
        """Return the reusable BM3D+Noise2SR image path for one source image."""
        stem = Path(image_name).stem
        return self.preprocessed_dir / f"{stem}_2_bm3d_noise2sr_preprocessed.png"

    def _compare_preprocess_candidates(self, image_name: str) -> List[Path]:
        """Find BM3D+Noise2SR outputs left by compare_preprocess_full.py."""
        stem = Path(image_name).stem
        filename = f"{stem}_2_bm3d_noise2sr_preprocessed.png"
        direct_candidates = [
            self.compare_preprocess_dir / filename,
            self.compare_preprocess_dir / f"{stem}_preprocessed" / filename,
        ]

        candidates = []
        seen = set()
        for path in direct_candidates:
            if path not in seen:
                candidates.append(path)
                seen.add(path)

        if any(path.exists() for path in candidates):
            return candidates

        if self.compare_preprocess_dir.exists():
            for path in self.compare_preprocess_dir.rglob(filename):
                if path not in seen:
                    candidates.append(path)
                    seen.add(path)

        return candidates

    @staticmethod
    def _normalize_bm3d_noise2sr_image(image: np.ndarray) -> np.ndarray:
        """Normalize cached/preprocessed images to HWC, 3-channel, uint8."""
        if image is None:
            raise ValueError("Input image is None")

        out = image
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        elif out.ndim == 3 and out.shape[2] == 1:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        elif out.ndim == 3 and out.shape[2] == 4:
            out = cv2.cvtColor(out, cv2.COLOR_BGRA2BGR)
        elif out.ndim != 3 or out.shape[2] != 3:
            raise ValueError(f"Unsupported preprocessed image shape: {out.shape}")

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

    def _load_bm3d_noise2sr_image(self, path: Path) -> Optional[np.ndarray]:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"  WARNING: Could not read cached BM3D+Noise2SR image: {path}")
            return None
        return self._normalize_bm3d_noise2sr_image(image)

    @staticmethod
    def _is_compatible_preprocessed_shape(source: np.ndarray, preprocessed: np.ndarray) -> bool:
        source_h, source_w = source.shape[:2]
        prep_h, prep_w = preprocessed.shape[:2]
        return (
            prep_h <= source_h
            and prep_w <= source_w
            and (source_h - prep_h) in (0, 1)
            and (source_w - prep_w) in (0, 1)
        )

    def _save_bm3d_noise2sr_image(
        self,
        image_name: str,
        image: np.ndarray,
        source: str,
    ) -> np.ndarray:
        preprocessed = self._normalize_bm3d_noise2sr_image(image)
        cache_path = self._bm3d_noise2sr_cache_path(image_name)

        if not cv2.imwrite(str(cache_path), preprocessed):
            raise IOError(f"Failed to save BM3D+Noise2SR image: {cache_path}")

        metadata = {
            "release_preprocessing": preprocessing_record(preprocessed.shape, None),
            "image_name": Path(image_name).stem,
            "method": "bm3d+noise2sr",
            "sigma_psd": 40,
            "clahe_applied": False,
            "source": source,
            "saved_path": str(cache_path),
            "shape": list(preprocessed.shape),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        metadata_path = cache_path.with_suffix(".json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return preprocessed

    def preprocess_image(self, image: np.ndarray, image_name: Optional[str] = None) -> np.ndarray:
        """
        BM3D + Noise2SR preprocessing for shape validation.

        Reuse order:
        1. shape_validation/preprocessed_bm3d_noise2sr cache
        2. compare_preprocess_full.py BM3D+Noise2SR output
        3. Fresh BM3D+Noise2SR run, saved for future reuse

        Args:
            image: Grayscale image
            image_name: Original image filename or stem, used for cache lookup

        Returns:
            BM3D+Noise2SR image normalized as HWC BGR uint8
        """
        if image is None:
            raise ValueError("Input image is None")

        if image_name:
            cache_path = self._bm3d_noise2sr_cache_path(image_name)
            if cache_path.exists() and preprocessing_record_matches(cache_path, image.shape, None):
                cached = self._load_bm3d_noise2sr_image(cache_path)
                if cached is not None and self._is_compatible_preprocessed_shape(image, cached):
                    print(f"  [CACHE] BM3D+Noise2SR: {cache_path.name}")
                    return cached
                if cached is not None:
                    print(f"  WARNING: Ignoring incompatible BM3D+Noise2SR cache: {cache_path}")

            for candidate in self._compare_preprocess_candidates(image_name):
                if not candidate.exists() or not preprocessing_record_matches(candidate, image.shape, None):
                    continue

                reused = self._load_bm3d_noise2sr_image(candidate)
                if reused is None:
                    continue
                if not self._is_compatible_preprocessed_shape(image, reused):
                    print(f"  WARNING: Ignoring incompatible compare_preprocess result: {candidate}")
                    continue

                print(f"  [REUSE] BM3D+Noise2SR from compare_preprocess: {candidate}")
                return self._save_bm3d_noise2sr_image(
                    image_name,
                    reused,
                    source=str(candidate),
                )

        # Run exactly BM3D + Noise2SR, without CLAHE, and save the result.
        preprocessed, _preproc_info = preprocess_with_config(
            image,
            sigma_psd=40,
            bm3d_enabled=True,
            noise2sr_enabled=True,
            clahe_enabled=False,
            verbose=True,
        )

        if not (_preproc_info.get("bm3d_applied") and _preproc_info.get("noise2sr_applied")):
            method = _preproc_info.get("denoising_method", "unknown")
            raise RuntimeError(
                f"Expected BM3D+Noise2SR preprocessing, but got '{method}'. "
                "Check BM3D/Noise2SR availability before saving a reusable cache."
            )
        if _preproc_info.get("clahe_applied"):
            raise RuntimeError("Shape validation cache must not include CLAHE.")

        if image_name:
            return self._save_bm3d_noise2sr_image(
                image_name,
                preprocessed,
                source="generated_by_shape_validation",
            )

        return self._normalize_bm3d_noise2sr_image(preprocessed)

    # =========================================================================
    # SAM2 INFERENCE (panalysis_debug.py reproduction)
    # =========================================================================

    def filter_background_masks(self, masks: List[Dict], image_shape: Tuple[int, int],
                                border_tolerance: int = 5) -> List[Dict]:
        """
        Remove background masks (4-border touching OR >90% area)

        Args:
            masks: SAM2 mask list
            image_shape: (height, width)
            border_tolerance: Border detection tolerance (pixels)

        Returns:
            Filtered mask list
        """
        return filter_sam_background_masks(
            masks,
            image_shape,
            border_tolerance=border_tolerance,
        )

    def filter_overlapping_masks(self, masks: List[Dict]) -> List[Dict]:
        """Centroid containment based overlap removal"""
        return filter_overlapping_masks_by_centroid(masks)

    def run_sam2_inference(self, image: np.ndarray) -> List[Dict]:
        """
        Run SAM2 inference with filtering

        Args:
            image: Preprocessed image

        Returns:
            List of mask dictionaries with 'segmentation' key
        """
        predictor = self.load_sam2_model()

        # Convert to RGB
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image

        # Generate masks
        masks = predictor.generate(image_rgb)

        # Filter background
        masks = self.filter_background_masks(masks, image.shape[:2])

        # Filter overlaps
        masks = self.filter_overlapping_masks(masks)

        return masks

    # =========================================================================
    # CLIP CLASSIFICATION (modules/shape_analysis.py reproduction)
    # =========================================================================

    def crop_and_zoom_particle(self, mask_image: np.ndarray, padding_ratio: float = 0.15,
                                target_size: int = 224) -> Image.Image:
        """
        Crop particle from mask and zoom to fill frame

        Args:
            mask_image: Particle image with black background
            padding_ratio: Proportion of bounding box to add as padding
            target_size: Final square image size (224 for CLIP)

        Returns:
            PIL.Image: Cropped and zoomed particle
        """
        # Convert to grayscale if needed
        if len(mask_image.shape) == 3:
            gray = cv2.cvtColor(mask_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = mask_image.copy()

        # Find particle bounding box
        coords = np.column_stack(np.where(gray > 0))
        if len(coords) == 0:
            # No particle found, return white image
            return Image.fromarray(np.ones((target_size, target_size), dtype=np.uint8) * 255)

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Calculate padding
        height = y_max - y_min + 1
        width = x_max - x_min + 1
        pad_h = int(height * padding_ratio)
        pad_w = int(width * padding_ratio)

        # Apply padding with boundary checks
        y_min = max(0, y_min - pad_h)
        y_max = min(mask_image.shape[0], y_max + pad_h)
        x_min = max(0, x_min - pad_w)
        x_max = min(mask_image.shape[1], x_max + pad_w)

        # Crop particle region
        if len(mask_image.shape) == 3:
            cropped = mask_image[y_min:y_max, x_min:x_max, :]
        else:
            cropped = mask_image[y_min:y_max, x_min:x_max]

        # Create square canvas with white background
        crop_h, crop_w = cropped.shape[:2]
        max_dim = max(crop_h, crop_w)

        if len(mask_image.shape) == 3:
            square_canvas = np.ones((max_dim, max_dim, mask_image.shape[2]), dtype=np.uint8) * 255
            y_offset = (max_dim - crop_h) // 2
            x_offset = (max_dim - crop_w) // 2
            square_canvas[y_offset:y_offset+crop_h, x_offset:x_offset+crop_w, :] = cropped
        else:
            square_canvas = np.ones((max_dim, max_dim), dtype=np.uint8) * 255
            y_offset = (max_dim - crop_h) // 2
            x_offset = (max_dim - crop_w) // 2
            square_canvas[y_offset:y_offset+crop_h, x_offset:x_offset+crop_w] = cropped

        # Resize to target size
        resized = cv2.resize(square_canvas, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)

        # Convert to RGB if grayscale
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

        return Image.fromarray(resized)

    def classify_shapes_with_clip(self, masks: List[Dict], original_image: np.ndarray,
                                  batch_size: int = 64, temperature: float = None,
                                  confidence_threshold: float = None) -> Tuple[List[str], List[float]]:
        """Use the same mask-defined crop and classifier as GUI and external cases."""
        from modules.shape_analysis import classify_shapes_with_clip
        model, preprocess, tokens = self.load_clip_model()
        shapes, confidences, _ = classify_shapes_with_clip(
            masks, original_image, self.shape_labels, self.clip_prompts,
            model, preprocess, tokens, self.device, batch_size=batch_size,
            temperature=temperature, confidence_threshold=confidence_threshold)
        return shapes, confidences

    # =========================================================================
    # ANNOTATION MANAGEMENT
    # =========================================================================

    def check_existing_annotations(self, image_paths: List[Path]) -> Tuple[List[Path], List[Path]]:
        """
        Check which images already have annotations

        Args:
            image_paths: List of image paths

        Returns:
            (annotated_images, unannotated_images)
        """
        annotated = []
        unannotated = []

        for img_path in image_paths:
            ann_file = self.annotations_dir / f"{img_path.stem}_shape.json"
            if not ann_file.exists():
                unannotated.append(img_path)
                continue
            try:
                with ann_file.open('r', encoding='utf-8') as handle:
                    annotation_payload = json.load(handle)
                annotation_config = annotation_payload.get(
                    'annotation_metadata', {}
                ).get('sam_config')
            except (OSError, json.JSONDecodeError, AttributeError):
                annotation_config = None
            if self._sam_config_matches(annotation_config):
                annotated.append(img_path)
            else:
                unannotated.append(img_path)
                print(
                    f"  [STALE] {ann_file.name} is not tied to the active SAM config; "
                    "annotation review is required"
                )

        print(f"\n?뱥 Annotation Status:")
        print(f"   Already annotated: {len(annotated)} images")
        print(f"   Need annotation: {len(unannotated)} images")

        return annotated, unannotated

    def save_annotation(
        self,
        image_name: str,
        particles: List[Dict],
        annotations_dir: Optional[Path] = None,
        metadata: Optional[Dict] = None,
    ):
        """
        Save annotation to JSON file

        Args:
            image_name: Image file name (without extension)
            particles: List of particle annotations
        """
        # Convert numpy types to native Python types for JSON serialization
        particles_serializable = []
        for p in particles:
            particle_dict = {}
            for key, value in p.items():
                # Convert numpy bool_ to Python bool
                if isinstance(value, np.bool_):
                    particle_dict[key] = bool(value)
                # Convert numpy int types to Python int
                elif isinstance(value, (np.integer, np.int64, np.int32)):
                    particle_dict[key] = int(value)
                # Convert numpy float types to Python float
                elif isinstance(value, (np.floating, np.float64, np.float32)):
                    particle_dict[key] = float(value)
                else:
                    particle_dict[key] = value
            particles_serializable.append(particle_dict)

        annotation = {
            "image_name": image_name,
            "timestamp": datetime.now().isoformat(),
            "particles": particles_serializable
        }
        if metadata:
            annotation["annotation_metadata"] = metadata

        destination = Path(annotations_dir) if annotations_dir is not None else self.annotations_dir
        destination.mkdir(parents=True, exist_ok=True)
        ann_file = destination / f"{image_name}_shape.json"
        with open(ann_file, 'w', encoding='utf-8') as f:
            json.dump(annotation, f, indent=2)

    def load_annotation(
        self,
        image_name: str,
        annotations_dir: Optional[Path] = None,
    ) -> Optional[Dict]:
        """
        Load annotation from JSON file

        Args:
            image_name: Image file name (without extension)

        Returns:
            Annotation dictionary or None
        """
        source_dir = Path(annotations_dir) if annotations_dir is not None else self.annotations_dir
        ann_file = source_dir / f"{image_name}_shape.json"
        if not ann_file.exists():
            return None

        with open(ann_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_predictions(self, image_name: str, masks: List[Dict], predictions: List[Dict],
                        image_shape: Tuple[int, int], output_dir: Optional[Path] = None,
                        prediction_metadata: Optional[Dict] = None,
                        sam_config: Optional[Dict] = None):
        """
        Save CLIP predictions and encoded masks to JSON

        Args:
            image_name: Image filename (without extension)
            masks: SAM2 mask dictionaries with 'segmentation' key
            predictions: CLIP prediction dictionaries
            image_shape: Original image shape (H, W)
        """
        # Encode masks and combine with predictions
        pred_data = []
        for mask_dict, pred in zip(masks, predictions):
            segmentation = mask_dict['segmentation']
            encoded_mask = self.encode_mask(segmentation)

            pred_data.append({
                'particle_id': pred['particle_id'],
                'pred_shape': pred['pred_shape'],
                'pred_confidence': pred['pred_confidence'],
                'mask': encoded_mask,
                'mask_shape': list(segmentation.shape)
            })

        # Save with image metadata
        output = {
            'image_name': image_name,
            'image_shape': list(image_shape),
            'shape_labels': list(self.shape_labels),
            'shape_descriptions': dict(zip(self.shape_labels, self.shape_descriptions)),
            'clip_prompts': dict(zip(self.shape_labels, self.clip_prompts)),
            'particles': pred_data,
        }
        if sam_config is not None:
            output['sam_config'] = dict(sam_config)
        if prediction_metadata:
            output['prediction_metadata'] = prediction_metadata

        destination = Path(output_dir) if output_dir is not None else self.predictions_dir
        destination.mkdir(parents=True, exist_ok=True)
        pred_file = destination / f"{image_name}_pred.json"
        with open(pred_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)

    def load_prediction(
        self,
        image_name: str,
        predictions_dir: Optional[Path] = None,
        require_active_sam_config: bool = False,
    ) -> Optional[Tuple[List[Dict], List[Dict]]]:
        """
        Load existing prediction file if available

        Args:
            image_name: Image filename (without extension)

        Returns:
            Tuple of (masks, predictions) or None if file doesn't exist
            - masks: List of SAM2 mask dictionaries with 'segmentation' key
            - predictions: List of prediction dictionaries
        """
        source_dir = Path(predictions_dir) if predictions_dir is not None else self.predictions_dir
        pred_file = source_dir / f"{image_name}_pred.json"
        if not pred_file.exists():
            return None

        try:
            with open(pred_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if require_active_sam_config:
                cached_sam_config = data.get('sam_config')
                if not isinstance(cached_sam_config, dict):
                    print(
                        f"  [STALE] {pred_file.name} has no SAM threshold provenance; "
                        "re-running SAM2+CLIP"
                    )
                    return None
                if not self._sam_config_matches(cached_sam_config):
                    print(
                        f"  [STALE] {pred_file.name} SAM config differs; "
                        "re-running SAM2+CLIP"
                    )
                    return None

            # Decode masks
            masks = []
            predictions = []

            for particle in data['particles']:
                # Decode mask
                mask_shape = tuple(particle['mask_shape'])
                segmentation = self.decode_mask(particle['mask'], mask_shape)

                # Reconstruct mask dict (similar to SAM2 output)
                mask_dict = {
                    'segmentation': segmentation,
                    'area': int(segmentation.sum())
                }
                masks.append(mask_dict)

                # Reconstruct prediction dict
                pred_dict = {
                    'particle_id': particle['particle_id'],
                    'pred_shape': particle['pred_shape'],
                    'pred_confidence': particle['pred_confidence']
                }
                predictions.append(pred_dict)

            return masks, predictions

        except Exception as e:
            print(f"  WARNING: Failed to load prediction for {image_name}: {e}")
            print(f"           Will re-run SAM2+CLIP instead")
            return None

    @staticmethod
    def _align_cached_masks_to_image(
        masks: List[Dict], image_shape: Tuple[int, int]
    ) -> Tuple[List[Dict], str]:
        """Crop or pad a one-pixel even-dimension difference without resizing masks."""
        target_h, target_w = image_shape
        aligned_masks = []
        alignment_notes = set()

        for mask_dict in masks:
            segmentation = np.asarray(mask_dict['segmentation'], dtype=bool)
            source_h, source_w = segmentation.shape
            delta_h = target_h - source_h
            delta_w = target_w - source_w
            if abs(delta_h) > 1 or abs(delta_w) > 1:
                raise ValueError(
                    "Cached mask shape is incompatible with the preprocessed image: "
                    f"mask={segmentation.shape}, image={image_shape}"
                )

            if source_h > target_h or source_w > target_w:
                segmentation = segmentation[:target_h, :target_w]
                alignment_notes.add('cropped_bottom_or_right_by_one_pixel')
            if segmentation.shape != (target_h, target_w):
                padded = np.zeros((target_h, target_w), dtype=bool)
                copy_h = min(target_h, segmentation.shape[0])
                copy_w = min(target_w, segmentation.shape[1])
                padded[:copy_h, :copy_w] = segmentation[:copy_h, :copy_w]
                segmentation = padded
                alignment_notes.add('padded_bottom_or_right_by_one_pixel')

            if not segmentation.any():
                raise ValueError("A cached mask became empty after shape alignment")
            aligned = dict(mask_dict)
            aligned['segmentation'] = np.ascontiguousarray(segmentation)
            aligned['area'] = int(segmentation.sum())
            aligned_masks.append(aligned)

        alignment = ','.join(sorted(alignment_notes)) if alignment_notes else 'none'
        return aligned_masks, alignment

    @staticmethod
    def _annotation_id_sets(annotation: Dict) -> Tuple[set, set]:
        """Return all and non-skipped particle IDs from one annotation file."""
        particles = annotation.get('particles', [])
        all_ids = {int(item['particle_id']) for item in particles}
        valid_ids = {
            int(item['particle_id'])
            for item in particles
            if not bool(item.get('skipped', False))
        }
        return all_ids, valid_ids

    @staticmethod
    def _mask_fingerprint(masks: List[Dict], predictions: List[Dict]) -> str:
        """Create a stable fingerprint for an ordered particle-ID/mask collection."""
        if len(masks) != len(predictions):
            raise ValueError(
                f"Cannot fingerprint {len(masks)} masks and {len(predictions)} predictions"
            )

        digest = hashlib.sha256()
        for mask, prediction in zip(masks, predictions):
            particle_id = int(prediction['particle_id'])
            segmentation = np.ascontiguousarray(mask['segmentation'], dtype=np.uint8)
            digest.update(particle_id.to_bytes(8, byteorder='little', signed=True))
            digest.update(np.asarray(segmentation.shape, dtype=np.int64).tobytes())
            digest.update(segmentation.tobytes())
        return digest.hexdigest()

    @staticmethod
    def _prediction_fingerprint(predictions: List[Dict]) -> str:
        """Create a stable fingerprint that changes when CLIP output changes."""
        digest = hashlib.sha256()
        for prediction in predictions:
            particle_id = int(prediction['particle_id'])
            pred_shape = str(prediction.get('pred_shape', ''))
            confidence = float(prediction.get('pred_confidence', 0.0))
            digest.update(particle_id.to_bytes(8, byteorder='little', signed=True))
            digest.update(pred_shape.encode('utf-8'))
            digest.update(np.float64(confidence).tobytes())
        return digest.hexdigest()

    @staticmethod
    def _annotation_fingerprint(annotation: Dict) -> str:
        """Create a stable fingerprint from the source particle annotations."""
        payload = json.dumps(
            annotation.get('particles', []),
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        )
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    @staticmethod
    def _prediction_id_mode(annotation: Dict, predictions: List[Dict]) -> Optional[str]:
        """Describe whether prediction IDs cover all or only valid annotation IDs."""
        prediction_ids = [int(item['particle_id']) for item in predictions]
        if len(prediction_ids) != len(set(prediction_ids)):
            return None

        all_ids, valid_ids = ShapeValidator._annotation_id_sets(annotation)
        prediction_id_set = set(prediction_ids)
        if prediction_id_set == all_ids:
            return 'all_annotation_ids'
        if valid_ids and prediction_id_set == valid_ids:
            return 'valid_annotation_ids_only'
        return None

    def _load_annotation_compatible_masks(
        self,
        image_name: str,
        annotation: Dict,
        additional_mask_source_dirs: Optional[List[Path]] = None,
    ) -> Optional[Dict]:
        """Find saved masks whose IDs are compatible with the manual annotation."""
        validation_root = self.output_dir.parent

        preferred_json_dirs = [Path(path) for path in additional_mask_source_dirs or []]
        preferred_json_dirs.extend(
            [
                validation_root / 'shape_validation_SEM' / 'predictions',
                validation_root / 'shape_validation_TEM' / 'predictions_wROD',
                validation_root / 'shape_validation_TEM' / 'predictions',
            ]
        )
        fallback_json_dirs = [
            self.output_dir / 'predictions_bf',
            self.predictions_dir,
        ]

        seen_dirs = set()

        def try_json_dirs(source_dirs: List[Path]) -> Optional[Dict]:
            for source_dir in source_dirs:
                source_dir = Path(source_dir)
                source_key = str(source_dir.resolve())
                if source_key in seen_dirs:
                    continue
                seen_dirs.add(source_key)

                source_file = source_dir / f'{image_name}_pred.json'
                if not source_file.exists():
                    continue
                loaded = self.load_prediction(image_name, source_dir)
                if loaded is None:
                    continue
                masks, predictions = loaded
                if len(masks) != len(predictions):
                    continue
                id_mode = self._prediction_id_mode(annotation, predictions)
                if id_mode is None:
                    continue
                return {
                    'masks': masks,
                    'particle_ids': [int(item['particle_id']) for item in predictions],
                    'source_kind': 'prediction_json',
                    'source_path': source_file,
                    'annotation_id_mode': id_mode,
                }
            return None

        compatible = try_json_dirs(preferred_json_dirs)
        if compatible is not None:
            return compatible

        all_ids, valid_ids = self._annotation_id_sets(annotation)
        pickle_dirs = [
            validation_root
            / 'size_validation'
            / 'segmentation_cache_overlap_logic_True_bm3d_only',
            validation_root
            / 'size_validation'
            / 'segmentation_cache_overlap_logic_True_bm3d_only_x',
        ]
        for source_dir in pickle_dirs:
            source_file = source_dir / f'{image_name}_sam2_masks.pkl'
            if not source_file.exists():
                continue
            try:
                with open(source_file, 'rb') as handle:
                    cached_items = pickle.load(handle)
            except Exception as exc:
                print(f"  WARNING: Could not read mask cache {source_file}: {exc}")
                continue
            if not isinstance(cached_items, list):
                continue

            if len(cached_items) == len(all_ids):
                particle_ids = sorted(all_ids)
                id_mode = 'all_annotation_ids'
            elif valid_ids and len(cached_items) == len(valid_ids):
                particle_ids = sorted(valid_ids)
                id_mode = 'valid_annotation_ids_only'
            else:
                continue

            masks = []
            cache_valid = True
            for item in cached_items:
                segmentation = item.get('segmentation') if isinstance(item, dict) else item
                segmentation = np.asarray(segmentation, dtype=bool)
                if segmentation.ndim != 2 or not segmentation.any():
                    cache_valid = False
                    break
                masks.append(
                    {
                        'segmentation': np.ascontiguousarray(segmentation),
                        'area': int(segmentation.sum()),
                    }
                )
            if not cache_valid:
                continue

            return {
                'masks': masks,
                'particle_ids': particle_ids,
                'source_kind': 'bm3d_only_mask_pickle',
                'source_path': source_file,
                'annotation_id_mode': id_mode,
            }

        return try_json_dirs(fallback_json_dirs)

    def review_prediction_mismatches(
        self,
        predictions_dir: Path,
        annotation_output_dir: Path,
        annotations_dir: Optional[Path] = None,
        image_stems: Optional[List[str]] = None,
    ) -> Dict:
        """Re-annotate only disagreements between saved predictions and saved GT."""
        predictions_dir = Path(predictions_dir).resolve()
        annotations_dir = Path(annotations_dir or self.annotations_dir).resolve()
        annotation_output_dir = Path(annotation_output_dir).resolve()

        if not predictions_dir.is_dir():
            raise FileNotFoundError(f'Prediction directory not found: {predictions_dir}')
        if not annotations_dir.is_dir():
            raise FileNotFoundError(f'Annotation directory not found: {annotations_dir}')
        if annotation_output_dir == annotations_dir:
            raise ValueError(
                'Mismatch-review output must differ from the source annotation directory.'
            )
        annotation_output_dir.mkdir(parents=True, exist_ok=True)

        prediction_suffix = '_pred.json'
        annotation_suffix = '_shape.json'
        prediction_stems = {
            path.name[:-len(prediction_suffix)]
            for path in predictions_dir.glob(f'*{prediction_suffix}')
        }
        annotation_stems = {
            path.name[:-len(annotation_suffix)]
            for path in annotations_dir.glob(f'*{annotation_suffix}')
        }
        if image_stems:
            requested = {Path(stem).stem for stem in image_stems}
            available = prediction_stems | annotation_stems
            missing_requested = sorted(requested - available)
            if missing_requested:
                raise FileNotFoundError(
                    'Requested image stems were not found: '
                    + ', '.join(missing_requested)
                )
            prediction_stems &= requested
            annotation_stems &= requested
        if not annotation_stems:
            raise ValueError(f'No annotation JSON files found in {annotations_dir}')

        compared_stems = sorted(annotation_stems & prediction_stems)
        annotation_only_stems = sorted(annotation_stems - prediction_stems)
        prediction_only_stems = sorted(prediction_stems - annotation_stems)

        print("\n" + "=" * 80)
        print("PREDICTION / GT MISMATCH REVIEW")
        print("=" * 80)
        print(f"Source annotations: {annotations_dir}")
        print(f"Source predictions: {predictions_dir}")
        print(f"Reviewed annotations: {annotation_output_dir}")
        print("No preprocessing, SAM2 inference, or CLIP inference will run.")
        print("Only saved prediction/GT disagreements and conflicting GT rows will be shown.")
        print("Use Skip Object when the displayed mask is not a particle.")
        if annotation_only_stems:
            print(
                "WARNING: Annotation images without saved predictions will be copied "
                "unchanged: " + ', '.join(annotation_only_stems)
            )
        if prediction_only_stems:
            print(
                "WARNING: Prediction images without GT cannot be reviewed: "
                + ', '.join(prediction_only_stems)
            )

        reviewed_now = 0
        already_reviewed = 0
        conflict_count = 0
        mismatch_count = 0
        unsupported_gt_count = 0
        review_target_count = 0
        unmatched_annotation_ids = 0
        unmatched_prediction_ids = 0
        reviewed_images = []

        for image_name in sorted(annotation_stems):
            source_annotation = self.load_annotation(
                image_name, annotations_dir=annotations_dir
            )
            if source_annotation is None:
                raise ValueError(f'Could not load source annotation for {image_name}')

            source_annotation_file = annotations_dir / f'{image_name}_shape.json'
            source_annotation_fingerprint = self._annotation_fingerprint(
                source_annotation
            )

            if image_name not in prediction_stems:
                self.save_annotation(
                    image_name,
                    source_annotation.get('particles', []),
                    annotations_dir=annotation_output_dir,
                    metadata={
                        'mode': 'existing_prediction_gt_mismatch_review',
                        'source_annotation': str(source_annotation_file.resolve()),
                        'source_annotation_fingerprint': source_annotation_fingerprint,
                        'prediction_available': False,
                        'reviewed_particle_ids': [],
                    },
                )
                continue

            loaded_prediction = self.load_prediction(image_name, predictions_dir)
            if loaded_prediction is None:
                raise ValueError(f'Could not load predictions for {image_name}')
            masks, predictions = loaded_prediction
            if len(masks) != len(predictions):
                raise ValueError(
                    f'Mask/prediction count mismatch for {image_name}: '
                    f'{len(masks)} vs {len(predictions)}'
                )

            prediction_ids = [int(item['particle_id']) for item in predictions]
            if len(prediction_ids) != len(set(prediction_ids)):
                raise ValueError(f'Duplicate prediction particle IDs in {image_name}')
            prediction_fingerprint = self._prediction_fingerprint(predictions)
            mask_fingerprint = self._mask_fingerprint(masks, predictions)

            previous_annotation = self.load_annotation(
                image_name, annotations_dir=annotation_output_dir
            )
            previously_reviewed_ids = set()
            previous_rows_by_id = {}
            if previous_annotation is not None:
                previous_metadata = previous_annotation.get('annotation_metadata', {})
            else:
                previous_metadata = {}
            previous_is_compatible = (
                previous_metadata.get('source_annotation_fingerprint')
                == source_annotation_fingerprint
                and previous_metadata.get('prediction_fingerprint')
                == prediction_fingerprint
                and previous_metadata.get('mask_fingerprint') == mask_fingerprint
            )
            if previous_is_compatible:
                previously_reviewed_ids = {
                    int(value)
                    for value in previous_metadata.get('reviewed_particle_ids', [])
                }
                for row in previous_annotation.get('particles', []):
                    previous_rows_by_id.setdefault(int(row['particle_id']), []).append(row)
                previously_reviewed_ids &= set(previous_rows_by_id)

            rows_by_id = {}
            for row in source_annotation.get('particles', []):
                particle_id = int(row['particle_id'])
                rows_by_id.setdefault(particle_id, []).append(row)

            annotation_ids = set(rows_by_id)
            prediction_id_set = set(prediction_ids)
            missing_prediction_ids = sorted(annotation_ids - prediction_id_set)
            prediction_without_gt_ids = sorted(prediction_id_set - annotation_ids)
            unmatched_annotation_ids += len(missing_prediction_ids)
            unmatched_prediction_ids += len(prediction_without_gt_ids)
            if missing_prediction_ids or prediction_without_gt_ids:
                details = []
                if missing_prediction_ids:
                    details.append(
                        'GT-only IDs=' + ','.join(map(str, missing_prediction_ids))
                    )
                if prediction_without_gt_ids:
                    details.append(
                        'prediction-only IDs='
                        + ','.join(map(str, prediction_without_gt_ids))
                    )
                print(f"  WARNING: {image_name}: {'; '.join(details)}")

            replacement_rows = {}
            review_masks = []
            review_predictions = []
            existing_gt_by_id = {}
            pending_review_ids = []

            for mask, prediction in zip(masks, predictions):
                particle_id = int(prediction['particle_id'])
                if particle_id not in rows_by_id:
                    continue
                pred_shape = _normalize_shape_label(
                    prediction.get('pred_shape'), self.shape_labels, is_prediction=True
                )
                rows = rows_by_id.get(particle_id, [])
                valid_rows = [
                    row for row in rows if not bool(row.get('skipped', False))
                ]
                skipped_rows = [
                    row for row in rows if bool(row.get('skipped', False))
                ]
                normalized_labels = []
                unsupported_labels = []
                for row in valid_rows:
                    label = _normalize_shape_label(
                        row.get('gt_shape'), self.shape_labels, is_prediction=False
                    )
                    if label is None:
                        unsupported_labels.append(str(row.get('gt_shape')))
                    elif label not in normalized_labels:
                        normalized_labels.append(label)

                needs_review = False
                display_labels = list(normalized_labels)
                if skipped_rows:
                    display_labels.append('<skipped>')
                if unsupported_labels:
                    display_labels.extend(
                        f'<unsupported: {label}>' for label in unsupported_labels
                    )
                existing_gt_by_id[particle_id] = ' / '.join(display_labels)

                if valid_rows:
                    label_conflict = (
                        len(normalized_labels) > 1
                        or bool(skipped_rows)
                        or bool(unsupported_labels)
                    )
                    if label_conflict:
                        conflict_count += 1
                        needs_review = True
                        if unsupported_labels:
                            unsupported_gt_count += 1
                    elif normalized_labels and normalized_labels[0] != pred_shape:
                        mismatch_count += 1
                        needs_review = True
                elif unsupported_labels:
                    unsupported_gt_count += 1
                    needs_review = True
                elif skipped_rows:
                    needs_review = False

                if needs_review:
                    review_target_count += 1
                    if particle_id in previously_reviewed_ids:
                        replacement_rows[particle_id] = dict(
                            previous_rows_by_id[particle_id][-1]
                        )
                        already_reviewed += 1
                    else:
                        pending_review_ids.append(particle_id)
                        review_masks.append(mask)
                        gui_prediction = dict(prediction)
                        gui_prediction['pred_shape'] = pred_shape
                        review_predictions.append(gui_prediction)

            reviewed_id_set = set(previously_reviewed_ids)
            if review_predictions:
                print(
                    f"\n[REVIEW] {image_name}: {len(review_predictions)} "
                    "prediction/GT disagreements"
                )
                image_path = self._bm3d_noise2sr_cache_path(image_name)
                image = self._load_bm3d_noise2sr_image(image_path)
                if image is None:
                    raise ValueError(f'Could not load review image: {image_path}')
                review_masks, _alignment = self._align_cached_masks_to_image(
                    review_masks, image.shape[:2]
                )
                corrections = self.run_annotation_gui(
                    image_name,
                    image,
                    review_masks,
                    review_predictions,
                    existing_gt_by_id=existing_gt_by_id,
                )
                correction_ids = [int(item['particle_id']) for item in corrections]
                if (
                    len(correction_ids) != len(set(correction_ids))
                    or set(correction_ids) != set(pending_review_ids)
                ):
                    raise RuntimeError(
                        f'Mismatch review for {image_name} is incomplete: '
                        f'expected IDs={sorted(pending_review_ids)}, '
                        f'got IDs={sorted(set(correction_ids))}'
                    )
                for correction in corrections:
                    particle_id = int(correction['particle_id'])
                    replacement_rows[particle_id] = correction
                    reviewed_id_set.add(particle_id)
                reviewed_now += len(corrections)
                reviewed_images.append(image_name)

            merged_particles = []
            replaced_ids = set()
            for source_row in source_annotation.get('particles', []):
                particle_id = int(source_row['particle_id'])
                if particle_id in replacement_rows:
                    if particle_id not in replaced_ids:
                        merged_particles.append(replacement_rows[particle_id])
                        replaced_ids.add(particle_id)
                    continue
                merged_particles.append(source_row)

            self.save_annotation(
                image_name,
                merged_particles,
                annotations_dir=annotation_output_dir,
                metadata={
                    'mode': 'existing_prediction_gt_mismatch_review',
                    'source_annotation': str(source_annotation_file.resolve()),
                    'source_annotation_fingerprint': source_annotation_fingerprint,
                    'prediction_file': str(
                        (predictions_dir / f'{image_name}_pred.json').resolve()
                    ),
                    'prediction_available': True,
                    'mask_fingerprint': mask_fingerprint,
                    'prediction_fingerprint': prediction_fingerprint,
                    'reviewed_particle_ids': sorted(reviewed_id_set),
                    'gt_only_particle_ids': missing_prediction_ids,
                    'prediction_only_particle_ids': prediction_without_gt_ids,
                },
            )

        manifest = {
            'mode': 'existing_prediction_gt_mismatch_review',
            'created_at': datetime.now().astimezone().isoformat(timespec='seconds'),
            'annotations_dir': str(annotations_dir),
            'predictions_dir': str(predictions_dir),
            'annotation_output_dir': str(annotation_output_dir),
            'annotation_image_count': len(annotation_stems),
            'compared_image_count': len(compared_stems),
            'annotation_images_without_prediction': annotation_only_stems,
            'prediction_images_without_annotation': prediction_only_stems,
            'review_target_count': review_target_count,
            'reviewed_now': reviewed_now,
            'previously_reviewed_reused': already_reviewed,
            'prediction_gt_mismatch_count': mismatch_count,
            'conflicting_gt_count': conflict_count,
            'unsupported_gt_count': unsupported_gt_count,
            'gt_only_particle_id_count': unmatched_annotation_ids,
            'prediction_only_particle_id_count': unmatched_prediction_ids,
            'reviewed_images_this_run': reviewed_images,
        }
        manifest_path = annotation_output_dir / 'mismatch_review_run.json'
        with open(manifest_path, 'w', encoding='utf-8') as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)

        print("\nMismatch review complete")
        print(f"  Review targets: {review_target_count}")
        print(f"  Newly reviewed particles: {reviewed_now}")
        print(f"  Previously reviewed particles reused: {already_reviewed}")
        print(f"  Conflicting duplicate GT entries found: {conflict_count}")
        print(f"  Prediction/GT mismatches found: {mismatch_count}")
        print(f"  GT-only particle IDs kept unchanged: {unmatched_annotation_ids}")
        print(f"  Prediction-only particle IDs not reviewed: {unmatched_prediction_ids}")
        print(f"  Reviewed images: {len(reviewed_images)}")
        print(f"  Manifest: {manifest_path}")
        return manifest

    def _skip_review_prediction_dirs(
        self,
        primary_predictions_dir: Path,
        additional_predictions_dirs: Optional[List[Path]] = None,
    ) -> List[Path]:
        """Return mask sources in the annotation-compatible priority order."""
        validation_root = self.output_dir.parent
        candidates = [Path(path) for path in additional_predictions_dirs or []]
        candidates.extend(
            [
                validation_root / 'shape_validation_SEM' / 'predictions',
                validation_root / 'shape_validation_TEM' / 'predictions_wROD',
                validation_root / 'shape_validation_TEM' / 'predictions',
                Path(primary_predictions_dir),
            ]
        )
        source_dirs = []
        seen = set()
        for candidate in candidates:
            resolved = candidate.resolve()
            key = str(resolved).lower()
            if key in seen or not resolved.is_dir():
                continue
            seen.add(key)
            source_dirs.append(resolved)
        return source_dirs

    def _skip_review_pickle_dirs(self) -> List[Path]:
        """Return legacy mask caches that retain annotation-era particle order."""
        size_root = self.output_dir.parent / 'size_validation'
        return [
            size_root / 'segmentation_cache_overlap_logic_True_bm3d_only',
            size_root / 'segmentation_cache_overlap_logic_True_bm3d_only_x',
        ]

    def _collect_skipped_annotation_records(
        self,
        predictions_dir: Path,
        image_stems: Optional[List[str]] = None,
        additional_predictions_dirs: Optional[List[Path]] = None,
    ) -> pd.DataFrame:
        """Collect skipped annotation candidates whose saved masks can be reviewed."""
        predictions_dir = Path(predictions_dir).resolve()
        prediction_source_dirs = self._skip_review_prediction_dirs(
            predictions_dir,
            additional_predictions_dirs=additional_predictions_dirs,
        )
        pickle_source_dirs = self._skip_review_pickle_dirs()
        annotation_suffix = '_shape.json'
        annotation_paths = sorted(self.annotations_dir.glob(f'*{annotation_suffix}'))
        available_stems = {
            path.name[:-len(annotation_suffix)] for path in annotation_paths
        }
        if image_stems:
            requested = {Path(value).stem for value in image_stems}
            missing = sorted(requested - available_stems)
            if missing:
                raise FileNotFoundError(
                    'Requested annotation image stems were not found: '
                    + ', '.join(missing)
                )
            annotation_paths = [
                path
                for path in annotation_paths
                if path.name[:-len(annotation_suffix)] in requested
            ]
        if not annotation_paths:
            raise ValueError(f'No annotation JSON files found in {self.annotations_dir}')

        skipped_rows = []

        for annotation_path in annotation_paths:
            image_name = annotation_path.name[:-len(annotation_suffix)]
            annotation = self.load_annotation(image_name)
            if annotation is None:
                raise ValueError(f'Could not load annotation: {annotation_path}')

            previously_reviewed_ids = set()
            annotation_metadata = annotation.get('annotation_metadata', {})
            mismatch_reviewed_ids = {
                int(value)
                for value in annotation_metadata.get('reviewed_particle_ids', [])
            }
            for review_entry in annotation_metadata.get('skip_review_history', []):
                previously_reviewed_ids.update(
                    int(value)
                    for value in review_entry.get('reviewed_particle_ids', [])
                )

            rows_by_id = {}
            particles = annotation.get('particles', [])
            for row in particles:
                particle_id = int(row['particle_id'])
                rows_by_id.setdefault(particle_id, []).append(row)

            prediction_file = None
            prediction_ids = set()
            for source_dir in prediction_source_dirs:
                candidate = source_dir / f'{image_name}_pred.json'
                if candidate.exists():
                    prediction_file = candidate
                    break
            if prediction_file is not None:
                try:
                    with open(prediction_file, 'r', encoding='utf-8') as handle:
                        prediction_payload = json.load(handle)
                    prediction_ids = {
                        int(row['particle_id'])
                        for row in prediction_payload.get('particles', [])
                        if row.get('mask') is not None and row.get('mask_shape') is not None
                    }
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    print(
                        f"  WARNING: Could not inspect saved masks for {image_name}: {exc}"
                    )

            skipped_ids = {
                particle_id
                for particle_id, rows in rows_by_id.items()
                if rows and all(bool(row.get('skipped', False)) for row in rows)
            }
            missing_prediction_ids = skipped_ids - prediction_ids
            pickle_file = None
            pickle_ids = set()
            if missing_prediction_ids:
                for source_dir in pickle_source_dirs:
                    candidate = source_dir / f'{image_name}_sam2_masks.pkl'
                    if not candidate.exists():
                        continue
                    try:
                        with open(candidate, 'rb') as handle:
                            cached_items = pickle.load(handle)
                    except Exception as exc:
                        print(f"  WARNING: Could not inspect mask cache {candidate}: {exc}")
                        continue
                    all_ids = sorted(rows_by_id)
                    if isinstance(cached_items, list) and len(cached_items) == len(all_ids):
                        pickle_file = candidate
                        pickle_ids = set(all_ids)
                        break

            for particle_id, rows in sorted(rows_by_id.items()):
                if not rows or not all(bool(row.get('skipped', False)) for row in rows):
                    continue
                skip_provenance = (
                    'latest_mismatch_review_final_skip'
                    if particle_id in mismatch_reviewed_ids
                    else 'preexisting_annotation_skip_flag'
                )
                if particle_id in prediction_ids:
                    mask_source_kind = 'prediction_json'
                    mask_source_path = prediction_file
                elif particle_id in pickle_ids:
                    mask_source_kind = 'mask_pickle'
                    mask_source_path = pickle_file
                else:
                    mask_source_kind = 'unavailable'
                    mask_source_path = None
                mask_available = mask_source_path is not None
                skipped_rows.append(
                    {
                        'Image': str(image_name),
                        'Particle_ID': int(particle_id),
                        'Annotation_Status': 'Skipped_non_particle_candidate',
                        'Skip_Provenance': skip_provenance,
                        'Saved_Mask_Available': bool(mask_available),
                        'Previously_Reviewed': particle_id in previously_reviewed_ids,
                        'Review_Status': (
                            'mask_unavailable'
                            if not mask_available
                            else (
                                'reviewed_still_skipped'
                                if particle_id in previously_reviewed_ids
                                else 'reviewable'
                            )
                        ),
                        'Mask_Source_Kind': mask_source_kind,
                        'Mask_Source_Path': (
                            str(mask_source_path) if mask_source_path is not None else ''
                        ),
                    }
                )

        skipped_columns = [
            'Image',
            'Particle_ID',
            'Annotation_Status',
            'Skip_Provenance',
            'Saved_Mask_Available',
            'Previously_Reviewed',
            'Review_Status',
            'Mask_Source_Kind',
            'Mask_Source_Path',
        ]
        return pd.DataFrame(skipped_rows, columns=skipped_columns)

    def run_skipped_mask_selection_gui(
        self,
        image_name: str,
        image: np.ndarray,
        masks: List[Dict],
        predictions: List[Dict],
        image_index: int,
        image_count: int,
        require_decision_for_all: bool = False,
        show_prediction: bool = True,
    ) -> Tuple[List[Dict], str, List[int]]:
        """Select skipped masks by clicking and annotate only the selected masks."""
        if not masks or len(masks) != len(predictions):
            raise ValueError(
                f'Invalid skipped-mask GUI input for {image_name}: '
                f'{len(masks)} masks vs {len(predictions)} predictions'
            )

        particle_ids = [int(item['particle_id']) for item in predictions]
        if len(particle_ids) != len(set(particle_ids)):
            raise ValueError(f'Duplicate skipped particle IDs for {image_name}')

        if image.ndim == 2:
            base_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 3:
            base_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f'Unsupported review image shape: {image.shape}')

        segmentations = [
            np.ascontiguousarray(mask['segmentation'], dtype=bool) for mask in masks
        ]
        mask_areas = [int(segmentation.sum()) for segmentation in segmentations]
        if any(area <= 0 for area in mask_areas):
            raise ValueError(f'Empty skipped mask found for {image_name}')

        corrections: Dict[int, Dict] = {}
        confirmed_skipped = set()
        state = {'selected_index': None, 'action': None}
        button_refs = []

        fig = plt.figure(figsize=(16, 10))
        grid = GridSpec(
            1,
            2,
            figure=fig,
            width_ratios=[1.45, 1.0],
            left=0.035,
            right=0.975,
            top=0.90,
            bottom=0.20,
            wspace=0.08,
        )
        ax_overview = fig.add_subplot(grid[0, 0])
        ax_detail = fig.add_subplot(grid[0, 1])

        def build_annotation(particle_index: int, shape: str) -> Dict:
            prediction = predictions[particle_index]
            segmentation = segmentations[particle_index]
            return {
                'particle_id': int(prediction['particle_id']),
                'gt_shape': shape,
                'pred_shape': prediction.get('pred_shape'),
                'pred_confidence': float(prediction.get('pred_confidence') or 0.0),
                'correct': bool(shape == prediction.get('pred_shape')),
                'border_touching': self.is_border_touching(segmentation),
            }

        def render():
            ax_overview.clear()
            ax_detail.clear()

            display = base_rgb.astype(np.float32).copy()
            selected_index = state['selected_index']
            for index, segmentation in enumerate(segmentations):
                particle_id = particle_ids[index]
                if index == selected_index:
                    color = np.array([255.0, 193.0, 7.0], dtype=np.float32)
                    alpha = 0.42
                elif particle_id in corrections:
                    color = np.array([42.0, 157.0, 143.0], dtype=np.float32)
                    alpha = 0.32
                elif particle_id in confirmed_skipped:
                    color = np.array([230.0, 126.0, 34.0], dtype=np.float32)
                    alpha = 0.28
                else:
                    color = np.array([31.0, 119.0, 180.0], dtype=np.float32)
                    alpha = 0.20
                display[segmentation] = (
                    display[segmentation] * (1.0 - alpha) + color * alpha
                )

            ax_overview.imshow(np.clip(display, 0, 255).astype(np.uint8))
            for index, segmentation in enumerate(segmentations):
                particle_id = particle_ids[index]
                if index == selected_index:
                    line_color = '#FFC107'
                    line_width = 2.8
                elif particle_id in corrections:
                    line_color = '#2A9D8F'
                    line_width = 2.2
                elif particle_id in confirmed_skipped:
                    line_color = '#E67E22'
                    line_width = 2.0
                else:
                    line_color = '#1F77B4'
                    line_width = 1.2
                contours, _ = cv2.findContours(
                    segmentation.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                for contour in contours:
                    points = contour[:, 0, :]
                    if len(points) < 2:
                        continue
                    closed = np.vstack([points, points[0]])
                    ax_overview.plot(
                        closed[:, 0],
                        closed[:, 1],
                        color=line_color,
                        linewidth=line_width,
                    )

            ax_overview.set_title(
                f'All masks: {len(masks)} | shape labels: {len(corrections)} | '
                f'confirmed non-particle: {len(confirmed_skipped)}'
            )
            ax_overview.axis('off')

            if selected_index is None:
                ax_detail.text(
                    0.5,
                    0.5,
                    'Click a skipped mask to inspect it',
                    ha='center',
                    va='center',
                    fontsize=15,
                    transform=ax_detail.transAxes,
                )
                ax_detail.set_title('No mask selected')
                ax_detail.axis('off')
            else:
                segmentation = segmentations[selected_index]
                prediction = predictions[selected_index]
                particle_id = particle_ids[selected_index]
                mask_image = np.zeros_like(image, dtype=np.uint8)
                mask_image[segmentation] = image[segmentation]
                particle_crop = self.crop_and_zoom_particle(mask_image)
                ax_detail.imshow(particle_crop)
                assigned_shape = corrections.get(particle_id, {}).get('gt_shape')
                if assigned_shape is not None:
                    status = f'Annotated now: {assigned_shape}'
                elif particle_id in confirmed_skipped:
                    status = 'Confirmed non-particle / keep skipped'
                else:
                    status = 'UNREVIEWED'
                if show_prediction:
                    prediction_text = (
                        f"Saved CLIP: {prediction.get('pred_shape')} "
                        f"({float(prediction.get('pred_confidence') or 0.0):.2%})"
                    )
                else:
                    prediction_text = 'Model prediction hidden during annotation'
                ax_detail.set_title(
                    f'Particle ID {particle_id} | {status}\n{prediction_text}'
                )
                ax_detail.axis('off')

            fig.suptitle(
                f'{image_name} | image {image_index}/{image_count}',
                fontsize=15,
                fontweight='bold',
            )
            fig.canvas.draw_idle()

        def select_mask(event):
            if event.inaxes is not ax_overview or event.button != 1:
                return
            if event.xdata is None or event.ydata is None:
                return
            x = int(np.floor(event.xdata + 0.5))
            y = int(np.floor(event.ydata + 0.5))
            if not (0 <= y < base_rgb.shape[0] and 0 <= x < base_rgb.shape[1]):
                return
            hits = [
                index
                for index, segmentation in enumerate(segmentations)
                if segmentation[y, x]
            ]
            if not hits:
                state['selected_index'] = None
            else:
                state['selected_index'] = min(hits, key=lambda index: mask_areas[index])
                particle_id = particle_ids[state['selected_index']]
                print(f'  Selected particle ID {particle_id}')
            render()

        def annotate_selected(shape: str):
            selected_index = state['selected_index']
            if selected_index is None:
                print('  Click a mask before selecting a shape.')
                return
            annotation = build_annotation(selected_index, shape)
            particle_id = int(annotation['particle_id'])
            corrections[particle_id] = annotation
            confirmed_skipped.discard(particle_id)
            print(f'  Particle ID {particle_id}: annotated as {shape}')
            render()

        def keep_selected_skipped():
            selected_index = state['selected_index']
            if selected_index is None:
                print('  Click a mask before changing its status.')
                return
            particle_id = particle_ids[selected_index]
            corrections.pop(particle_id, None)
            confirmed_skipped.add(particle_id)
            print(f'  Particle ID {particle_id}: confirmed non-particle / remains skipped')
            render()

        def finish(action: str):
            if require_decision_for_all:
                decided_ids = set(corrections) | confirmed_skipped
                missing_ids = sorted(set(particle_ids) - decided_ids)
                if missing_ids:
                    print(
                        '  Review every mask before saving. Undecided particle IDs: '
                        + ', '.join(str(value) for value in missing_ids)
                    )
                    return
            state['action'] = action
            plt.close(fig)

        def on_close(_event):
            if state['action'] is None:
                state['action'] = 'cancel'

        def on_key(event):
            if event.key in {str(index) for index in range(1, len(self.shape_labels) + 1)}:
                annotate_selected(self.shape_labels[int(event.key) - 1])
            elif event.key in {'delete', 'backspace'}:
                keep_selected_skipped()
            elif event.key in {'enter', 'right'}:
                finish('next')
            elif event.key in {'q', 'escape'}:
                finish('stop')

        shape_button_width = 0.135
        for index, label in enumerate(self.shape_labels):
            button_axis = fig.add_axes(
                [0.03 + index * 0.145, 0.105, shape_button_width, 0.055]
            )
            button = Button(button_axis, f'[{index + 1}] {label}', color='lightgray')
            button.on_clicked(lambda _event, value=label: annotate_selected(value))
            button_refs.append(button)

        keep_axis = fig.add_axes([0.78, 0.105, 0.18, 0.055])
        keep_button = Button(keep_axis, 'Keep Skipped', color='#F4A261')
        keep_button.on_clicked(lambda _event: keep_selected_skipped())
        button_refs.append(keep_button)

        next_axis = fig.add_axes([0.55, 0.025, 0.19, 0.055])
        next_button = Button(next_axis, 'Save & Next Image', color='#A8DADC')
        next_button.on_clicked(lambda _event: finish('next'))
        button_refs.append(next_button)

        stop_axis = fig.add_axes([0.77, 0.025, 0.19, 0.055])
        stop_button = Button(stop_axis, 'Save & Stop', color='#E9C46A')
        stop_button.on_clicked(lambda _event: finish('stop'))
        button_refs.append(stop_button)

        fig.canvas.mpl_connect('button_press_event', select_mask)
        fig.canvas.mpl_connect('key_press_event', on_key)
        fig.canvas.mpl_connect('close_event', on_close)
        render()
        plt.show(block=True)

        action = state['action'] or 'cancel'
        prediction_order = {
            particle_id: index for index, particle_id in enumerate(particle_ids)
        }
        selected_annotations = sorted(
            corrections.values(),
            key=lambda item: prediction_order[int(item['particle_id'])],
        )
        return selected_annotations, action, sorted(confirmed_skipped)

    def review_skipped_annotations(
        self,
        predictions_dir: Optional[Path] = None,
        image_stems: Optional[List[str]] = None,
        additional_predictions_dirs: Optional[List[Path]] = None,
        include_previously_reviewed: bool = True,
        require_decision_for_all: bool = False,
        show_predictions_in_gui: bool = True,
    ) -> Dict:
        """Click skipped masks to annotate selected objects in the base annotations."""
        predictions_dir = Path(predictions_dir or self.predictions_dir).resolve()
        if not predictions_dir.is_dir():
            raise FileNotFoundError(f'Prediction directory not found: {predictions_dir}')

        skipped_df = self._collect_skipped_annotation_records(
            predictions_dir=predictions_dir,
            image_stems=image_stems,
            additional_predictions_dirs=additional_predictions_dirs,
        )
        if skipped_df.empty:
            reviewable_df = skipped_df.copy()
            unavailable_df = skipped_df.copy()
        else:
            reviewable_mask = skipped_df['Saved_Mask_Available'].astype(bool).copy()
            if not include_previously_reviewed:
                reviewable_mask &= ~skipped_df['Previously_Reviewed']
            reviewable_df = skipped_df[reviewable_mask].copy()
            unavailable_df = skipped_df[~skipped_df['Saved_Mask_Available']].copy()
        previously_reviewed_count = int(
            skipped_df['Previously_Reviewed'].sum()
        ) if not skipped_df.empty else 0
        print("\n" + "=" * 80)
        print("SKIPPED OBJECT REVIEW")
        print("=" * 80)
        print("No preprocessing, SAM2 inference, or CLIP inference will run.")
        print(f"Skipped candidates: {len(skipped_df)}")
        print(f"Reviewable saved masks: {len(reviewable_df)}")
        print(f"Previously reviewed and still skipped: {previously_reviewed_count}")
        print(f"Masks unavailable: {len(unavailable_df)}")
        print(f"Annotations will be updated in place: {self.annotations_dir.resolve()}")
        if not unavailable_df.empty:
            unavailable_labels = [
                f"{row.Image}:{int(row.Particle_ID)}"
                for row in unavailable_df.itertuples(index=False)
            ]
            print("Unavailable saved masks: " + ", ".join(unavailable_labels))

        reviewed_ids = []
        reviewed_images = []
        displayed_images = []
        stopped_early = False
        review_image_names = sorted(reviewable_df['Image'].unique())
        for image_position, image_name in enumerate(review_image_names, start=1):
            image_review_df = reviewable_df[
                reviewable_df['Image'] == image_name
            ].copy()
            target_ids = set(image_review_df['Particle_ID'].astype(int))
            annotation = self.load_annotation(image_name)
            if annotation is None:
                raise ValueError(f'Could not load annotation for {image_name}')
            annotation_rows_by_id = {}
            for row in annotation.get('particles', []):
                annotation_rows_by_id.setdefault(int(row['particle_id']), []).append(row)

            saved_masks_by_id = {}
            source_paths_used = []
            for source_path_text in image_review_df['Mask_Source_Path'].unique():
                source_path = Path(source_path_text)
                source_rows = image_review_df[
                    image_review_df['Mask_Source_Path'] == source_path_text
                ]
                source_ids = set(source_rows['Particle_ID'].astype(int))
                source_kind = str(source_rows.iloc[0]['Mask_Source_Kind'])
                source_paths_used.append(str(source_path.resolve()))
                if source_kind == 'prediction_json':
                    loaded_prediction = self.load_prediction(
                        image_name, source_path.parent
                    )
                    if loaded_prediction is None:
                        raise ValueError(f'Could not load saved masks: {source_path}')
                    masks, predictions = loaded_prediction
                    for mask, prediction in zip(masks, predictions):
                        particle_id = int(prediction['particle_id'])
                        if particle_id in source_ids:
                            saved_masks_by_id[particle_id] = mask
                elif source_kind == 'mask_pickle':
                    with open(source_path, 'rb') as handle:
                        cached_items = pickle.load(handle)
                    all_ids = sorted(annotation_rows_by_id)
                    if not isinstance(cached_items, list) or len(cached_items) != len(all_ids):
                        raise ValueError(
                            f'Cached mask count changed for {image_name}: '
                            f'{len(cached_items) if isinstance(cached_items, list) else "invalid"} '
                            f'vs {len(all_ids)} annotation IDs'
                        )
                    for particle_id, item in zip(all_ids, cached_items):
                        if particle_id not in source_ids:
                            continue
                        segmentation = (
                            item.get('segmentation') if isinstance(item, dict) else item
                        )
                        segmentation = np.asarray(segmentation, dtype=bool)
                        if segmentation.ndim != 2 or not segmentation.any():
                            raise ValueError(
                                f'Invalid cached mask for {image_name} particle {particle_id}'
                            )
                        saved_masks_by_id[particle_id] = {
                            'segmentation': np.ascontiguousarray(segmentation),
                            'area': int(segmentation.sum()),
                        }
                else:
                    raise ValueError(
                        f'Unsupported skip-review mask source: {source_kind}'
                    )

            missing_ids = sorted(target_ids - set(saved_masks_by_id))
            if missing_ids:
                raise ValueError(
                    f'Saved masks changed while preparing {image_name}; missing IDs={missing_ids}'
                )

            ordered_ids = sorted(target_ids)
            review_masks = [saved_masks_by_id[particle_id] for particle_id in ordered_ids]
            review_predictions = []
            for particle_id in ordered_ids:
                source_row = annotation_rows_by_id[particle_id][-1]
                review_predictions.append(
                    {
                        'particle_id': particle_id,
                        'pred_shape': _normalize_shape_label(
                            source_row.get('pred_shape'),
                            self.shape_labels,
                            is_prediction=True,
                        ),
                        'pred_confidence': float(
                            source_row.get('pred_confidence') or 0.0
                        ),
                    }
                )
            image_path = self._bm3d_noise2sr_cache_path(image_name)
            image = self._load_bm3d_noise2sr_image(image_path)
            if image is None:
                raise ValueError(f'Could not load review image: {image_path}')
            review_masks, alignment = self._align_cached_masks_to_image(
                review_masks, image.shape[:2]
            )
            print(f"\n[REVIEW] {image_name}: {len(ordered_ids)} skipped candidates")
            corrections, action, confirmed_skipped_ids = self.run_skipped_mask_selection_gui(
                image_name=image_name,
                image=image,
                masks=review_masks,
                predictions=review_predictions,
                image_index=image_position,
                image_count=len(review_image_names),
                require_decision_for_all=require_decision_for_all,
                show_prediction=show_predictions_in_gui,
            )
            if action == 'cancel':
                print('  Review window closed without saving the current image.')
                stopped_early = True
                break

            correction_ids = [int(row['particle_id']) for row in corrections]
            if (
                len(correction_ids) != len(set(correction_ids))
                or not set(correction_ids).issubset(target_ids)
            ):
                raise RuntimeError(
                    f'Skip review returned invalid IDs for {image_name}: '
                    f'targets={ordered_ids}, selected={sorted(set(correction_ids))}'
                )

            replacement_by_id = {
                int(row['particle_id']): dict(row) for row in corrections
            }
            decision_ids = sorted(set(replacement_by_id) | set(confirmed_skipped_ids))
            displayed_images.append(image_name)
            if decision_ids:
                merged_particles = []
                inserted = set()
                for row in annotation.get('particles', []):
                    particle_id = int(row['particle_id'])
                    if particle_id in replacement_by_id:
                        if particle_id not in inserted:
                            merged_particles.append(replacement_by_id[particle_id])
                            inserted.add(particle_id)
                        continue
                    merged_particles.append(row)

                metadata = dict(annotation.get('annotation_metadata', {}))
                history = list(metadata.get('skip_review_history', []))
                history.append(
                    {
                        'reviewed_at': datetime.now().astimezone().isoformat(
                            timespec='seconds'
                        ),
                        'mode': 'click_selected_skipped_masks',
                        'mask_sources': source_paths_used,
                        'displayed_skipped_particle_ids': ordered_ids,
                        'reviewed_particle_ids': decision_ids,
                        'annotated_particle_ids': sorted(replacement_by_id),
                        'kept_skipped_particle_ids': sorted(confirmed_skipped_ids),
                        'prediction_displayed_in_gui': bool(show_predictions_in_gui),
                        'mask_alignment': alignment,
                    }
                )
                metadata['skip_review_history'] = history
                if replacement_by_id:
                    metadata['manual_labels_modified'] = True
                    if not show_predictions_in_gui:
                        metadata['zero_classifiable_review_blinded_to_predictions'] = True
                self.save_annotation(
                    image_name,
                    merged_particles,
                    annotations_dir=self.annotations_dir,
                    metadata=metadata,
                )
                reviewed_ids.extend(
                    (image_name, particle_id) for particle_id in correction_ids
                )
                reviewed_images.append(image_name)
                print(
                    f'  Saved {len(correction_ids)} shape annotation(s); '
                    f'confirmed {len(confirmed_skipped_ids)} non-particle mask(s).'
                )
            else:
                print('  No masks selected; annotation file left unchanged.')

            if action == 'stop':
                stopped_early = image_position < len(review_image_names)
                break

        manifest = {
            'mode': 'click_select_skipped_masks_in_place',
            'created_at': datetime.now().astimezone().isoformat(timespec='seconds'),
            'annotations_dir': str(self.annotations_dir.resolve()),
            'predictions_dir': str(predictions_dir),
            'prediction_mask_source_dirs': [
                str(path)
                for path in self._skip_review_prediction_dirs(
                    predictions_dir,
                    additional_predictions_dirs=additional_predictions_dirs,
                )
            ],
            'reviewable_before_review': len(reviewable_df),
            'previously_reviewed_still_skipped_before_review': previously_reviewed_count,
            'mask_unavailable_before_review': len(unavailable_df),
            'displayed_image_count': len(displayed_images),
            'reviewed_image_count': len(reviewed_images),
            'reviewed_object_count': len(reviewed_ids),
            'stopped_early': bool(stopped_early),
            'displayed_images': displayed_images,
            'reviewed_images': reviewed_images,
        }
        manifest_path = self.results_dir / 'shape_skip_review_manifest.json'
        with open(manifest_path, 'w', encoding='utf-8') as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
        print(f"  Review manifest: {manifest_path}")
        return manifest

    def run_prompt_reevaluation(
        self,
        prediction_output_dir: Path,
        annotation_override_dir: Path,
        evaluation_output_dir: Path,
        additional_mask_source_dirs: Optional[List[Path]] = None,
        image_stems: Optional[List[str]] = None,
    ) -> Dict:
        """Reclassify compatible masks and annotate only regenerated fallback masks."""
        prediction_output_dir = Path(prediction_output_dir)
        annotation_override_dir = Path(annotation_override_dir)
        evaluation_output_dir = Path(evaluation_output_dir)
        prediction_output_dir.mkdir(parents=True, exist_ok=True)
        annotation_override_dir.mkdir(parents=True, exist_ok=True)
        evaluation_output_dir.mkdir(parents=True, exist_ok=True)

        dataset_images = sorted(
            path
            for path in self.dataset_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {'.jpg', '.png', '.bmp'}
        )
        if image_stems:
            requested = {Path(stem).stem for stem in image_stems}
            dataset_images = [path for path in dataset_images if path.stem in requested]
            missing_requested = sorted(requested - {path.stem for path in dataset_images})
            if missing_requested:
                raise FileNotFoundError(
                    'Requested Dataset_shape images were not found: '
                    + ', '.join(missing_requested)
                )
        if not dataset_images:
            raise FileNotFoundError(f'No input images found in {self.dataset_dir}')

        missing_preprocessed = [
            image_path.stem
            for image_path in dataset_images
            if not self._bm3d_noise2sr_cache_path(image_path.stem).exists()
        ]
        if missing_preprocessed:
            raise FileNotFoundError(
                'BM3D+Noise2SR preprocessed images are missing for: '
                + ', '.join(missing_preprocessed)
            )

        print("\n" + "=" * 80)
        print("NEW-PROMPT SHAPE REEVALUATION")
        print("=" * 80)
        print(f"Images: {len(dataset_images)}")
        print(f"Predictions: {prediction_output_dir}")
        print(f"Fallback annotations: {annotation_override_dir}")
        print(f"Results: {evaluation_output_dir}")
        print("Compatible saved masks will run through CLIP only.")
        print("Only incompatible or missing masks will run SAM2 and request annotation.")

        summary_rows = []
        source_counts = Counter()
        fallback_images = []
        reused_fallback_annotations = []

        for image_path in tqdm(dataset_images, desc='New-prompt reevaluation'):
            image_name = image_path.stem
            preprocessed_path = self._bm3d_noise2sr_cache_path(image_name)
            preprocessed = self._load_bm3d_noise2sr_image(preprocessed_path)
            if preprocessed is None:
                raise ValueError(f'Could not load preprocessed image: {preprocessed_path}')

            base_annotation = self.load_annotation(image_name)
            mask_source = None
            if base_annotation is not None:
                mask_source = self._load_annotation_compatible_masks(
                    image_name,
                    base_annotation,
                    additional_mask_source_dirs=additional_mask_source_dirs,
                )

            requires_annotation = mask_source is None
            source_kind = None
            source_path = None
            annotation_id_mode = None

            if mask_source is not None:
                masks = mask_source['masks']
                particle_ids = mask_source['particle_ids']
                source_kind = mask_source['source_kind']
                source_path = Path(mask_source['source_path'])
                annotation_id_mode = mask_source['annotation_id_mode']
            else:
                output_file = prediction_output_dir / f'{image_name}_pred.json'
                override_annotation = self.load_annotation(
                    image_name, annotations_dir=annotation_override_dir
                )
                reusable_fallback = None
                if output_file.exists() and override_annotation is not None:
                    try:
                        with open(output_file, 'r', encoding='utf-8') as handle:
                            output_json = json.load(handle)
                        output_metadata = output_json.get('prediction_metadata', {})
                        if output_metadata.get('mask_source_kind') == 'fallback_sam':
                            loaded_output = self.load_prediction(
                                image_name, predictions_dir=prediction_output_dir
                            )
                            if loaded_output is not None:
                                old_masks, old_predictions = loaded_output
                                stored_fingerprint = override_annotation.get(
                                    'annotation_metadata', {}
                                ).get('mask_fingerprint')
                                actual_fingerprint = self._mask_fingerprint(
                                    old_masks, old_predictions
                                )
                                if (
                                    stored_fingerprint == actual_fingerprint
                                    and self._prediction_id_mode(
                                        override_annotation, old_predictions
                                    ) == 'all_annotation_ids'
                                ):
                                    reusable_fallback = (old_masks, old_predictions)
                    except Exception as exc:
                        print(
                            f"  WARNING: Ignoring incomplete fallback cache for "
                            f"{image_name}: {exc}"
                        )

                if reusable_fallback is not None:
                    masks, old_predictions = reusable_fallback
                    particle_ids = [
                        int(item['particle_id']) for item in old_predictions
                    ]
                    source_kind = 'fallback_sam_reused'
                    source_path = output_file
                    annotation_id_mode = 'override_annotation_ids'
                    requires_annotation = False
                    reused_fallback_annotations.append(image_name)
                else:
                    print(
                        f"\n[FALLBACK] {image_name}: no annotation-compatible saved masks; "
                        "running SAM2 on the cached BM3D+Noise2SR image."
                    )
                    masks = self.run_sam2_inference(preprocessed)
                    if not masks:
                        raise RuntimeError(f'No fallback SAM2 masks found for {image_name}')
                    particle_ids = list(range(len(masks)))
                    source_kind = 'fallback_sam'
                    source_path = preprocessed_path
                    annotation_id_mode = 'new_annotation_required'
                    fallback_images.append(image_name)

            masks, alignment = self._align_cached_masks_to_image(
                masks, preprocessed.shape[:2]
            )
            if len(masks) != len(particle_ids):
                raise ValueError(
                    f'Mask/ID count mismatch for {image_name}: '
                    f'{len(masks)} vs {len(particle_ids)}'
                )

            shapes, confidences = self.classify_shapes_with_clip(masks, preprocessed)
            predictions = [
                {
                    'particle_id': particle_id,
                    'pred_shape': shape,
                    'pred_confidence': confidence,
                }
                for particle_id, shape, confidence in zip(
                    particle_ids, shapes, confidences
                )
            ]
            mask_fingerprint = self._mask_fingerprint(masks, predictions)
            prediction_metadata = {
                'mode': 'new_prompt_reevaluation',
                'created_at': datetime.now().astimezone().isoformat(timespec='seconds'),
                'clip_model': 'ViT-L/14@336px',
                'preprocessed_image': str(preprocessed_path.resolve()),
                'mask_source_kind': (
                    'fallback_sam'
                    if source_kind in {'fallback_sam', 'fallback_sam_reused'}
                    else source_kind
                ),
                'mask_source': str(Path(source_path).resolve()),
                'mask_alignment': alignment,
                'annotation_id_mode': annotation_id_mode,
                'mask_fingerprint': mask_fingerprint,
                'preprocessing_rerun': False,
                'sam_rerun': source_kind == 'fallback_sam',
            }
            self.save_predictions(
                image_name,
                masks,
                predictions,
                preprocessed.shape[:2],
                output_dir=prediction_output_dir,
                prediction_metadata=prediction_metadata,
            )

            if requires_annotation:
                print(
                    f"\n[ANNOTATION REQUIRED] {image_name}: "
                    f"{len(masks)} regenerated masks"
                )
                annotations = self.run_annotation_gui(
                    image_path.name, preprocessed, masks, predictions
                )
                annotation_ids = [int(item['particle_id']) for item in annotations]
                expected_ids = set(particle_ids)
                if (
                    len(annotation_ids) != len(set(annotation_ids))
                    or set(annotation_ids) != expected_ids
                ):
                    raise RuntimeError(
                        f'Annotation for {image_name} is incomplete: '
                        f'expected IDs={sorted(expected_ids)}, got IDs={sorted(set(annotation_ids))}'
                    )
                self.save_annotation(
                    image_name,
                    annotations,
                    annotations_dir=annotation_override_dir,
                    metadata={
                        'mode': 'fallback_mask_annotation',
                        'mask_fingerprint': mask_fingerprint,
                        'prediction_file': str(
                            (prediction_output_dir / f'{image_name}_pred.json').resolve()
                        ),
                        'preprocessed_image': str(preprocessed_path.resolve()),
                    },
                )

            source_counts[source_kind] += 1
            for prediction in predictions:
                summary_rows.append(
                    {
                        'image_name': image_name,
                        'particle_id': prediction['particle_id'],
                        'pred_shape': prediction['pred_shape'],
                        'pred_confidence': prediction['pred_confidence'],
                        'mask_source_kind': source_kind,
                        'mask_source': str(Path(source_path).resolve()),
                        'annotation_id_mode': annotation_id_mode,
                        'mask_alignment': alignment,
                    }
                )

        summary_df = pd.DataFrame(summary_rows)
        summary_path = prediction_output_dir / 'prediction_summary.csv'
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        prediction_manifest = {
            'mode': 'new_prompt_reevaluation',
            'created_at': datetime.now().astimezone().isoformat(timespec='seconds'),
            'dataset_dir': str(self.dataset_dir.resolve()),
            'prediction_output_dir': str(prediction_output_dir.resolve()),
            'annotation_override_dir': str(annotation_override_dir.resolve()),
            'evaluation_output_dir': str(evaluation_output_dir.resolve()),
            'shape_labels': list(self.shape_labels),
            'shape_descriptions': dict(zip(self.shape_labels, self.shape_descriptions)),
            'clip_prompts': dict(zip(self.shape_labels, self.clip_prompts)),
            'image_count': len(dataset_images),
            'particle_count': len(summary_df),
            'mask_source_image_counts': dict(source_counts),
            'fallback_sam_images': fallback_images,
            'reused_fallback_annotation_images': reused_fallback_annotations,
        }
        manifest_path = prediction_output_dir / 'prediction_run.json'
        with open(manifest_path, 'w', encoding='utf-8') as handle:
            json.dump(prediction_manifest, handle, indent=2, ensure_ascii=False)

        print("\nPrediction phase complete")
        print(f"  Source counts: {dict(source_counts)}")
        print(f"  New fallback annotations: {fallback_images}")
        print(f"  Prediction manifest: {manifest_path}")

        evaluation = self.evaluate_prediction_directory(
            predictions_dir=prediction_output_dir,
            evaluation_output_dir=evaluation_output_dir,
            annotation_override_dir=annotation_override_dir,
            image_stems=[path.stem for path in dataset_images],
        )
        return {
            'prediction_manifest_path': manifest_path,
            'prediction_summary_path': summary_path,
            'fallback_images': fallback_images,
            'evaluation': evaluation,
        }

    def run_clip_prediction_only(
        self,
        prediction_output_dir: Path,
        additional_mask_source_dirs: Optional[List[Path]] = None,
        image_stems: Optional[List[str]] = None,
    ) -> Dict:
        """Reclassify cached SAM masks with CLIP without preprocessing or SAM inference."""
        prediction_output_dir = Path(prediction_output_dir)
        prediction_output_dir.mkdir(parents=True, exist_ok=True)

        dataset_images = sorted(
            path
            for path in self.dataset_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {'.jpg', '.png', '.bmp'}
        )
        if image_stems:
            requested = {Path(stem).stem for stem in image_stems}
            dataset_images = [path for path in dataset_images if path.stem in requested]
            missing_requested = sorted(requested - {path.stem for path in dataset_images})
            if missing_requested:
                raise FileNotFoundError(
                    "Requested Dataset_shape images were not found: "
                    + ', '.join(missing_requested)
                )
        if not dataset_images:
            raise FileNotFoundError(f"No input images found in {self.dataset_dir}")

        mask_source_dirs = [self.predictions_dir]
        for source_dir in additional_mask_source_dirs or []:
            source_path = Path(source_dir)
            if source_path not in mask_source_dirs:
                mask_source_dirs.append(source_path)
        validation_root = self.output_dir.parent
        for fallback_dir in [
            validation_root / 'shape_validation_TEM' / 'predictions',
            validation_root / 'shape_validation_TEM' / 'predictions_wROD',
        ]:
            if fallback_dir not in mask_source_dirs:
                mask_source_dirs.append(fallback_dir)

        jobs = []
        preflight_errors = []
        for image_path in dataset_images:
            image_name = image_path.stem
            preprocessed_path = self._bm3d_noise2sr_cache_path(image_name)
            if not preprocessed_path.exists():
                preflight_errors.append(
                    f"{image_name}: missing preprocessed cache {preprocessed_path}"
                )
                continue

            mask_source_path = next(
                (
                    source_dir / f'{image_name}_pred.json'
                    for source_dir in mask_source_dirs
                    if (source_dir / f'{image_name}_pred.json').exists()
                ),
                None,
            )
            if mask_source_path is None:
                preflight_errors.append(
                    f"{image_name}: no mask-bearing prediction JSON found"
                )
                continue
            jobs.append((image_path, preprocessed_path, mask_source_path))

        if preflight_errors:
            details = '\n'.join(f"  - {error}" for error in preflight_errors)
            raise RuntimeError(
                "CLIP-only prediction requires an existing preprocessed image and "
                f"cached masks for every image:\n{details}"
            )

        print("\n" + "=" * 80)
        print("CLIP-ONLY SHAPE RECLASSIFICATION")
        print("=" * 80)
        print(f"Images: {len(jobs)}")
        print(f"Output: {prediction_output_dir}")
        print("No preprocessing or SAM inference will run.")

        summary_rows = []
        source_counts = Counter()
        alignment_counts = Counter()
        for image_path, preprocessed_path, mask_source_path in tqdm(
            jobs, desc="CLIP-only predictions"
        ):
            image_name = image_path.stem
            preprocessed = self._load_bm3d_noise2sr_image(preprocessed_path)
            if preprocessed is None:
                raise ValueError(f"Could not load preprocessed cache: {preprocessed_path}")

            loaded = self.load_prediction(image_name, mask_source_path.parent)
            if loaded is None:
                raise ValueError(f"Could not load cached masks: {mask_source_path}")
            masks, source_predictions = loaded
            masks, alignment = self._align_cached_masks_to_image(
                masks, preprocessed.shape[:2]
            )
            if len(masks) != len(source_predictions):
                raise AssertionError(
                    f"Mask/prediction count mismatch for {image_name}: "
                    f"{len(masks)} vs {len(source_predictions)}"
                )

            shapes, confidences = self.classify_shapes_with_clip(masks, preprocessed)
            predictions = [
                {
                    'particle_id': source_prediction['particle_id'],
                    'pred_shape': shape,
                    'pred_confidence': confidence,
                }
                for source_prediction, shape, confidence in zip(
                    source_predictions, shapes, confidences
                )
            ]
            metadata = {
                'mode': 'clip_only_reclassification',
                'created_at': datetime.now().astimezone().isoformat(timespec='seconds'),
                'clip_model': 'ViT-L/14@336px',
                'preprocessed_image': str(preprocessed_path.resolve()),
                'mask_source': str(mask_source_path.resolve()),
                'mask_alignment': alignment,
                'preprocessing_rerun': False,
                'sam_rerun': False,
            }
            self.save_predictions(
                image_name,
                masks,
                predictions,
                preprocessed.shape[:2],
                output_dir=prediction_output_dir,
                prediction_metadata=metadata,
            )

            source_counts[str(mask_source_path.parent.resolve())] += 1
            alignment_counts[alignment] += 1
            for prediction in predictions:
                summary_rows.append(
                    {
                        'image_name': image_name,
                        'particle_id': prediction['particle_id'],
                        'pred_shape': prediction['pred_shape'],
                        'pred_confidence': prediction['pred_confidence'],
                        'mask_source': str(mask_source_path.resolve()),
                        'mask_alignment': alignment,
                    }
                )

        summary_df = pd.DataFrame(
            summary_rows,
            columns=[
                'image_name',
                'particle_id',
                'pred_shape',
                'pred_confidence',
                'mask_source',
                'mask_alignment',
            ],
        )
        summary_path = prediction_output_dir / 'prediction_summary.csv'
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')

        manifest = {
            'mode': 'clip_only_reclassification',
            'created_at': datetime.now().astimezone().isoformat(timespec='seconds'),
            'dataset_dir': str(self.dataset_dir.resolve()),
            'output_dir': str(prediction_output_dir.resolve()),
            'clip_model': 'ViT-L/14@336px',
            'shape_labels': list(self.shape_labels),
            'shape_descriptions': dict(zip(self.shape_labels, self.shape_descriptions)),
            'clip_prompts': dict(zip(self.shape_labels, self.clip_prompts)),
            'image_count': len(jobs),
            'particle_count': len(summary_df),
            'mask_source_image_counts': dict(source_counts),
            'mask_alignment_image_counts': dict(alignment_counts),
            'preprocessing_rerun': False,
            'sam_rerun': False,
            'outputs': {
                'per_image_predictions': '*_pred.json',
                'summary': summary_path.name,
            },
        }
        manifest_path = prediction_output_dir / 'prediction_run.json'
        with open(manifest_path, 'w', encoding='utf-8') as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)

        print(f"[SUCCESS] CLIP-only predictions: {len(jobs)} images, {len(summary_df)} particles")
        print(f"          Summary: {summary_path}")
        print(f"          Manifest: {manifest_path}")
        return {
            'output_dir': prediction_output_dir,
            'summary_path': summary_path,
            'manifest_path': manifest_path,
            'image_count': len(jobs),
            'particle_count': len(summary_df),
        }

    @staticmethod
    def _pixel_fingerprint(image: np.ndarray) -> str:
        """Hash normalized pixels, including shape and dtype."""
        array = np.ascontiguousarray(image)
        digest = hashlib.sha256()
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(str(array.dtype).encode('ascii'))
        digest.update(array.tobytes())
        return digest.hexdigest()

    def _clip_prompt_fingerprint(self) -> str:
        payload = json.dumps(
            self.clip_prompts,
            ensure_ascii=False,
            separators=(',', ':'),
        )
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    @staticmethod
    def _read_json_dict(path: Path) -> Dict:
        if not path.is_file():
            return {}
        try:
            with path.open('r', encoding='utf-8') as handle:
                payload = json.load(handle)
            return payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _select_fixed_preprocessed_image(
        self,
        image_path: Path,
        external_preprocessed_dir: Optional[Path],
        reuse_preprocessed_only: bool,
    ) -> Tuple[np.ndarray, Dict]:
        """Prefer the verified size cache, then the existing shape cache."""
        source_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if source_image is None:
            raise IOError(f"Could not read Dataset_shape image: {image_path}")

        filename = self._bm3d_noise2sr_cache_path(image_path.stem).name
        candidates = []
        if external_preprocessed_dir is not None:
            candidates.append(
                ('external_verified_cache', Path(external_preprocessed_dir) / filename)
            )
        candidates.append(
            ('shape_cache', self._bm3d_noise2sr_cache_path(image_path.stem))
        )

        rejected = []
        for source_kind, candidate in candidates:
            if not candidate.is_file():
                continue
            preprocessed = self._load_bm3d_noise2sr_image(candidate)
            if preprocessed is None:
                rejected.append(f'{candidate}: unreadable')
                continue
            if not self._is_compatible_preprocessed_shape(source_image, preprocessed):
                rejected.append(
                    f'{candidate}: incompatible shape {preprocessed.shape[:2]} '
                    f'for source {source_image.shape[:2]}'
                )
                continue

            metadata_path = candidate.with_suffix('.json')
            metadata = self._read_json_dict(metadata_path)
            method = str(metadata.get('method', '')).lower().replace(' ', '')
            if method and method not in {'bm3d+noise2sr', 'bm3d_noise2sr'}:
                rejected.append(f'{candidate}: recorded method={metadata.get("method")!r}')
                continue
            if bool(metadata.get('clahe_applied', False)):
                rejected.append(f'{candidate}: CLAHE is recorded as enabled')
                continue
            recorded_epochs = metadata.get('noise2sr_epochs')
            if recorded_epochs is not None and int(recorded_epochs) != 1500:
                rejected.append(
                    f'{candidate}: recorded Noise2SR epochs={recorded_epochs}, expected 1500'
                )
                continue

            return preprocessed, {
                'source_kind': source_kind,
                'path': str(candidate.resolve()),
                'metadata_path': (
                    str(metadata_path.resolve()) if metadata_path.is_file() else None
                ),
                'method': metadata.get('method', 'bm3d+noise2sr'),
                'noise2sr_epochs': (
                    int(recorded_epochs) if recorded_epochs is not None else 1500
                ),
                'noise2sr_epochs_provenance': (
                    'recorded_in_cache_metadata'
                    if recorded_epochs is not None
                    else 'inferred_from_modules.preprocessing_code_default'
                ),
                'clahe_applied': bool(metadata.get('clahe_applied', False)),
                'pixel_sha256': self._pixel_fingerprint(preprocessed),
            }

        if reuse_preprocessed_only:
            detail = '; '.join(rejected) if rejected else 'no cache file found'
            raise FileNotFoundError(
                f'No usable cached BM3D+Noise2SR image for {image_path.stem}: {detail}. '
                'Cache-only mode forbids a fresh Noise2SR run.'
            )

        preprocessed = self.preprocess_image(source_image, image_path.stem)
        return preprocessed, {
            'source_kind': 'generated_by_shape_validation',
            'path': str(self._bm3d_noise2sr_cache_path(image_path.stem).resolve()),
            'metadata_path': str(
                self._bm3d_noise2sr_cache_path(image_path.stem).with_suffix('.json').resolve()
            ),
            'method': 'bm3d+noise2sr',
            'noise2sr_epochs': 1500,
            'noise2sr_epochs_provenance': 'modules.preprocessing_code_default',
            'clahe_applied': False,
            'pixel_sha256': self._pixel_fingerprint(preprocessed),
        }

    def _load_external_fixed_masks(
        self,
        image_name: str,
        preprocessed: np.ndarray,
        external_mask_cache_dir: Optional[Path],
        external_preprocessed_dir: Optional[Path],
    ) -> Optional[Tuple[List[Dict], Dict]]:
        """Reuse a size mask cache only when thresholds and input pixels match."""
        if external_mask_cache_dir is None or external_preprocessed_dir is None:
            return None
        external_mask_cache_dir = Path(external_mask_cache_dir)
        external_preprocessed_dir = Path(external_preprocessed_dir)
        expected_tag = (
            f'sam_p{round(self.pred_iou_thresh * 100):03d}_'
            f's{round(self.stability_score_thresh * 100):03d}'
        )
        if expected_tag not in external_mask_cache_dir.name:
            raise ValueError(
                'External SAM cache directory does not identify the active thresholds: '
                f'expected tag {expected_tag!r} in {external_mask_cache_dir.name!r}'
            )

        mask_path = external_mask_cache_dir / f'{image_name}_sam2_masks.pkl'
        preprocessed_path = (
            external_preprocessed_dir
            / f'{image_name}_2_bm3d_noise2sr_preprocessed.png'
        )
        if not mask_path.is_file() or not preprocessed_path.is_file():
            return None

        external_pixels = self._load_bm3d_noise2sr_image(preprocessed_path)
        if external_pixels is None:
            return None
        if (
            external_pixels.shape != preprocessed.shape
            or not np.array_equal(external_pixels, preprocessed)
        ):
            return None

        with mask_path.open('rb') as handle:
            cached = pickle.load(handle)
        if not isinstance(cached, (list, tuple)):
            raise TypeError(f'Unsupported mask cache payload in {mask_path}')

        masks = []
        for item in cached:
            segmentation = item.get('segmentation') if isinstance(item, dict) else item
            segmentation = np.asarray(segmentation, dtype=bool)
            if segmentation.ndim != 2:
                raise ValueError(f'Invalid cached mask shape in {mask_path}: {segmentation.shape}')
            masks.append(
                {
                    'segmentation': np.ascontiguousarray(segmentation),
                    'area': int(segmentation.sum()),
                }
            )
        masks, mask_alignment = self._align_cached_masks_to_image(
            masks, preprocessed.shape[:2]
        )
        return masks, {
            'kind': 'reused_size_sam2_cache',
            'mask_cache_path': str(mask_path.resolve()),
            'mask_input_path': str(preprocessed_path.resolve()),
            'mask_input_pixel_sha256': self._pixel_fingerprint(external_pixels),
            'mask_shape_alignment': mask_alignment,
        }

    def _load_resumable_fixed_prediction(
        self,
        image_name: str,
        predictions_dir: Path,
        preprocessing_sha256: str,
    ) -> Optional[Tuple[List[Dict], List[Dict]]]:
        prediction_path = Path(predictions_dir) / f'{image_name}_pred.json'
        payload = self._read_json_dict(prediction_path)
        metadata = payload.get('prediction_metadata', {})
        if not isinstance(metadata, dict):
            return None
        if metadata.get('preprocessing_pixel_sha256') != preprocessing_sha256:
            return None
        if metadata.get('clip_prompt_sha256') != self._clip_prompt_fingerprint():
            return None
        return self.load_prediction(
            image_name,
            predictions_dir=predictions_dir,
            require_active_sam_config=True,
        )

    @staticmethod
    def _mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        rows, cols = np.where(mask)
        if len(rows) == 0:
            return None
        return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1

    @classmethod
    def _mask_iou_matrix(cls, old_masks: List[Dict], new_masks: List[Dict]) -> np.ndarray:
        matrix = np.zeros((len(old_masks), len(new_masks)), dtype=np.float64)
        old_arrays = [np.asarray(item['segmentation'], dtype=bool) for item in old_masks]
        new_arrays = [np.asarray(item['segmentation'], dtype=bool) for item in new_masks]
        old_areas = [int(item.sum()) for item in old_arrays]
        new_areas = [int(item.sum()) for item in new_arrays]
        old_boxes = [cls._mask_bbox(item) for item in old_arrays]
        new_boxes = [cls._mask_bbox(item) for item in new_arrays]

        for old_index, (old, old_area, old_box) in enumerate(
            zip(old_arrays, old_areas, old_boxes)
        ):
            if old_box is None:
                continue
            oy0, oy1, ox0, ox1 = old_box
            for new_index, (new, new_area, new_box) in enumerate(
                zip(new_arrays, new_areas, new_boxes)
            ):
                if new_box is None:
                    continue
                ny0, ny1, nx0, nx1 = new_box
                y0, y1 = max(oy0, ny0), min(oy1, ny1)
                x0, x1 = max(ox0, nx0), min(ox1, nx1)
                if y0 >= y1 or x0 >= x1:
                    continue
                intersection = int(np.logical_and(
                    old[y0:y1, x0:x1], new[y0:y1, x0:x1]
                ).sum())
                if intersection == 0:
                    continue
                union = old_area + new_area - intersection
                matrix[old_index, new_index] = intersection / union if union else 0.0
        return matrix

    @classmethod
    def _hungarian_mask_matches(
        cls,
        old_masks: List[Dict],
        new_masks: List[Dict],
        iou_threshold: float,
    ) -> List[Tuple[int, int, float]]:
        """Maximize valid-match cardinality, then total mask IoU."""
        if not old_masks or not new_masks:
            return []
        iou = cls._mask_iou_matrix(old_masks, new_masks)
        size = max(iou.shape)
        score = np.zeros((size, size), dtype=np.float64)
        valid = iou >= float(iou_threshold)
        score[:iou.shape[0], :iou.shape[1]] = np.where(valid, 1.0 + iou, 0.0)
        rows, cols = linear_sum_assignment(-score)
        matches = []
        for old_index, new_index in zip(rows, cols):
            if old_index >= iou.shape[0] or new_index >= iou.shape[1]:
                continue
            value = float(iou[old_index, new_index])
            if value >= iou_threshold:
                matches.append((int(old_index), int(new_index), value))
        return matches

    @staticmethod
    def _canonical_source_annotation_rows(annotation: Optional[Dict]) -> Tuple[Dict[int, Dict], int]:
        rows_by_id = {}
        for row in (annotation or {}).get('particles', []):
            try:
                particle_id = int(row['particle_id'])
            except (KeyError, TypeError, ValueError):
                continue
            rows_by_id.setdefault(particle_id, []).append(row)

        canonical = {}
        duplicates = 0
        for particle_id, rows in rows_by_id.items():
            duplicates += max(0, len(rows) - 1)
            valid_rows = [row for row in rows if not bool(row.get('skipped', False))]
            canonical[particle_id] = valid_rows[-1] if valid_rows else rows[-1]
        return canonical, duplicates

    def _transfer_annotations_by_hungarian(
        self,
        image_name: str,
        new_masks: List[Dict],
        new_predictions: List[Dict],
        annotation_output_dir: Path,
        iou_threshold: float,
    ) -> Dict:
        """Transfer existing manual labels from old masks to fixed-setting masks."""
        source_annotation = self.load_annotation(image_name)
        source_prediction = self.load_prediction(image_name, self.predictions_dir)
        old_masks = []
        old_predictions = []
        old_alignment = 'none'
        if source_prediction is not None:
            old_masks, old_predictions = source_prediction
            try:
                old_masks, old_alignment = self._align_cached_masks_to_image(
                    old_masks,
                    new_masks[0]['segmentation'].shape
                    if new_masks else tuple(
                        self._read_json_dict(
                            self.predictions_dir / f'{image_name}_pred.json'
                        ).get('image_shape', (0, 0))
                    ),
                )
            except ValueError:
                old_masks, old_predictions = [], []
                old_alignment = 'incompatible_source_masks_excluded'

        old_ids = [int(item['particle_id']) for item in old_predictions]
        source_rows, duplicate_rows = self._canonical_source_annotation_rows(
            source_annotation
        )
        matches = self._hungarian_mask_matches(
            old_masks, new_masks, iou_threshold=iou_threshold
        )
        match_by_new = {
            new_index: (old_index, match_iou)
            for old_index, new_index, match_iou in matches
        }

        transferred = []
        transferred_valid = 0
        for new_index, (new_mask, prediction) in enumerate(
            zip(new_masks, new_predictions)
        ):
            particle_id = int(prediction['particle_id'])
            row = {
                'particle_id': particle_id,
                'gt_shape': None,
                'pred_shape': prediction['pred_shape'],
                'pred_confidence': float(prediction['pred_confidence']),
                'correct': None,
                'border_touching': self.is_border_touching(
                    new_mask['segmentation']
                ),
                'skipped': True,
                'annotation_transfer_status': 'unmatched_new_mask',
            }
            matched = match_by_new.get(new_index)
            if matched is not None:
                old_index, match_iou = matched
                source_particle_id = old_ids[old_index]
                row['source_particle_id'] = source_particle_id
                row['transfer_mask_iou'] = float(match_iou)
                source_row = source_rows.get(source_particle_id)
                if source_row is not None:
                    skipped = bool(source_row.get('skipped', False))
                    gt_shape = source_row.get('gt_shape')
                    normalized_gt = _normalize_shape_label(
                        gt_shape, self.shape_labels, is_prediction=False
                    )
                    if not skipped and normalized_gt is not None:
                        row.update(
                            {
                                'gt_shape': normalized_gt,
                                'skipped': False,
                                'correct': bool(
                                    normalized_gt == prediction['pred_shape']
                                ),
                                'annotation_transfer_status': 'valid_label_transferred',
                            }
                        )
                        transferred_valid += 1
                    else:
                        row['gt_shape'] = normalized_gt
                        row['annotation_transfer_status'] = 'source_annotation_skipped'
            transferred.append(row)

        source_annotation_ids = set(source_rows)
        old_id_set = set(old_ids)
        alignment = {
            'method': 'Hungarian assignment on old-vs-new binary-mask IoU',
            'mask_iou_threshold': float(iou_threshold),
            'new_mask_count': len(new_masks),
            'old_mask_count': len(old_masks),
            'hungarian_matched_pairs': len(matches),
            'transferred_valid_labels': transferred_valid,
            'new_unmatched_masks': len(new_masks) - len(matches),
            'old_unmatched_masks': len(old_masks) - len(matches),
            'source_annotation_gt_only_ids': len(source_annotation_ids - old_id_set),
            'source_duplicate_annotation_rows_collapsed': duplicate_rows,
            'source_mask_shape_alignment': old_alignment,
        }
        metadata = {
            'mode': 'fixed_sam_setting_hungarian_label_transfer',
            'sam_config': dict(self.sam_config),
            'blinded_to_predictions': False,
            'manual_labels_modified': False,
            'source_annotation_path': str(
                (self.annotations_dir / f'{image_name}_shape.json').resolve()
            ),
            'source_prediction_path': str(
                (self.predictions_dir / f'{image_name}_pred.json').resolve()
            ),
            'source_annotation_fingerprint': (
                self._annotation_fingerprint(source_annotation)
                if source_annotation is not None else None
            ),
            'alignment': alignment,
        }
        self.save_annotation(
            image_name,
            transferred,
            annotations_dir=annotation_output_dir,
            metadata=metadata,
        )
        return alignment

    def run_fixed_setting_validation(
        self,
        prediction_output_dir: Path,
        annotation_output_dir: Path,
        evaluation_output_dir: Path,
        external_mask_cache_dir: Optional[Path] = None,
        external_preprocessed_dir: Optional[Path] = None,
        image_stems: Optional[List[str]] = None,
        reuse_preprocessed_only: bool = True,
        alignment_iou_threshold: float = 0.50,
        plan_only: bool = False,
    ) -> Dict:
        """Run/resume fixed SAM2 Shape validation without opening annotation GUI."""
        if not 0.0 <= alignment_iou_threshold <= 1.0:
            raise ValueError('alignment_iou_threshold must be between 0 and 1')
        prediction_output_dir = Path(prediction_output_dir).resolve()
        annotation_output_dir = Path(annotation_output_dir).resolve()
        evaluation_output_dir = Path(evaluation_output_dir).resolve()
        prediction_output_dir.mkdir(parents=True, exist_ok=True)
        if not plan_only:
            annotation_output_dir.mkdir(parents=True, exist_ok=True)

        allowed_suffixes = {'.png', '.jpg', '.jpeg', '.bmp'}
        image_paths = sorted(
            path for path in self.dataset_dir.iterdir()
            if path.is_file() and path.suffix.lower() in allowed_suffixes
        )
        if image_stems:
            selected = {Path(stem).stem for stem in image_stems}
            image_paths = [path for path in image_paths if path.stem in selected]
            missing = sorted(selected - {path.stem for path in image_paths})
            if missing:
                raise FileNotFoundError(
                    'Requested Dataset_shape stems were not found: ' + ', '.join(missing)
                )

        counts = Counter()
        per_image = []
        for image_path in tqdm(image_paths, desc='Fixed-setting Shape validation'):
            preprocessed, preprocessing_info = self._select_fixed_preprocessed_image(
                image_path,
                external_preprocessed_dir=external_preprocessed_dir,
                reuse_preprocessed_only=reuse_preprocessed_only,
            )
            counts[f"preprocessing_{preprocessing_info['source_kind']}"] += 1
            preprocessing_sha = preprocessing_info['pixel_sha256']

            fixed_prediction = self._load_resumable_fixed_prediction(
                image_path.stem,
                prediction_output_dir,
                preprocessing_sha,
            )
            mask_source = None
            if fixed_prediction is not None:
                masks, predictions = fixed_prediction
                counts['fixed_predictions_reused'] += 1
                mask_source = {'kind': 'resumed_fixed_prediction'}
            else:
                external_masks = self._load_external_fixed_masks(
                    image_path.stem,
                    preprocessed,
                    external_mask_cache_dir=external_mask_cache_dir,
                    external_preprocessed_dir=external_preprocessed_dir,
                )
                if external_masks is not None:
                    masks, mask_source = external_masks
                    counts['fixed_sam_masks_reused'] += 1
                else:
                    masks, predictions = [], []
                    counts['sam2_inference_required'] += 1
                    mask_source = {'kind': 'sam2_generated_for_shape'}

                if not plan_only:
                    if not masks and mask_source['kind'] == 'sam2_generated_for_shape':
                        masks = self.run_sam2_inference(preprocessed)
                    if masks:
                        shapes, confidences = self.classify_shapes_with_clip(
                            masks, preprocessed
                        )
                    else:
                        shapes, confidences = [], []
                    predictions = [
                        {
                            'particle_id': index,
                            'pred_shape': shape,
                            'pred_confidence': confidence,
                        }
                        for index, (shape, confidence) in enumerate(
                            zip(shapes, confidences)
                        )
                    ]
                    self.save_predictions(
                        image_path.stem,
                        masks,
                        predictions,
                        preprocessed.shape[:2],
                        output_dir=prediction_output_dir,
                        prediction_metadata={
                            'mode': 'fixed_sam_setting_shape_validation',
                            'sam_model': 'SAM 2.1 Hiera Large',
                            'preprocessing': preprocessing_info,
                            'preprocessing_pixel_sha256': preprocessing_sha,
                            'mask_source': mask_source,
                            'clip_model': 'ViT-L/14@336px',
                            'clip_prompt_sha256': self._clip_prompt_fingerprint(),
                            'created_at': datetime.now().astimezone().isoformat(
                                timespec='seconds'
                            ),
                        },
                        sam_config=self.sam_config,
                    )

            image_record = {
                'image_name': image_path.stem,
                'preprocessing_source': preprocessing_info['source_kind'],
                'preprocessing_path': preprocessing_info['path'],
                'preprocessing_pixel_sha256': preprocessing_sha,
                'mask_source': mask_source['kind'],
            }
            if not plan_only:
                alignment = self._transfer_annotations_by_hungarian(
                    image_path.stem,
                    masks,
                    predictions,
                    annotation_output_dir=annotation_output_dir,
                    iou_threshold=alignment_iou_threshold,
                )
                image_record['alignment'] = alignment
            per_image.append(image_record)

        run_manifest = {
            'mode': 'fixed_sam_setting_shape_validation_plan' if plan_only else (
                'fixed_sam_setting_shape_validation'
            ),
            'created_at': datetime.now().astimezone().isoformat(timespec='seconds'),
            'dataset_dir': str(self.dataset_dir.resolve()),
            'sam_model': 'SAM 2.1 Hiera Large',
            'sam_config': dict(self.sam_config),
            'preprocessing': {
                'method': 'BM3D + Noise2SR',
                'bm3d_sigma_255': 40,
                'noise2sr_epochs': 1500,
                'clahe_applied': False,
                'cache_only': bool(reuse_preprocessed_only),
            },
            'external_mask_cache_dir': (
                str(Path(external_mask_cache_dir).resolve())
                if external_mask_cache_dir is not None else None
            ),
            'external_preprocessed_dir': (
                str(Path(external_preprocessed_dir).resolve())
                if external_preprocessed_dir is not None else None
            ),
            'alignment': {
                'method': 'Hungarian assignment on old-vs-new binary-mask IoU',
                'mask_iou_threshold': float(alignment_iou_threshold),
            },
            'counts': dict(counts),
            'images': per_image,
        }

        if plan_only:
            print('\nFIXED-SETTING SHAPE PLAN')
            print(f"  Dataset images: {len(image_paths)}")
            print(
                '  Verified Size preprocessing reused: '
                f"{counts['preprocessing_external_verified_cache']}"
            )
            print(
                '  Existing Shape preprocessing reused: '
                f"{counts['preprocessing_shape_cache']}"
            )
            print(
                '  Verified fixed SAM masks reusable: '
                f"{counts['fixed_sam_masks_reused']}"
            )
            print(
                '  Completed fixed predictions reusable: '
                f"{counts['fixed_predictions_reused']}"
            )
            print(f"  SAM2 inference still required: {counts['sam2_inference_required']}")
            required_stems = [
                item['image_name']
                for item in per_image
                if item['mask_source'] == 'sam2_generated_for_shape'
            ]
            if required_stems:
                print('  SAM2-required image stems: ' + ', '.join(required_stems))
            print('  Fresh BM3D/Noise2SR runs: 0')
            return run_manifest

        manifest_path = prediction_output_dir / 'fixed_setting_run.json'
        with manifest_path.open('w', encoding='utf-8') as handle:
            json.dump(run_manifest, handle, indent=2, ensure_ascii=False)

        evaluation = self.evaluate_prediction_directory(
            predictions_dir=prediction_output_dir,
            evaluation_output_dir=evaluation_output_dir,
            annotation_override_dir=annotation_output_dir,
            image_stems=[path.stem for path in image_paths],
            common_ids_only=False,
            comparison_policy=(
                'Hungarian old-to-new mask-IoU label transfer at IoU >= '
                f'{alignment_iou_threshold:.2f}; strict new prediction/annotation ID alignment'
            ),
        )
        run_manifest['manifest_path'] = str(manifest_path)
        run_manifest['evaluation'] = {
            'excel_path': str(evaluation['excel_path']),
            'manifest_path': str(evaluation['manifest_path']),
        }
        return run_manifest

    def evaluate_prediction_directory(
        self,
        predictions_dir: Path,
        evaluation_output_dir: Path,
        annotation_override_dir: Optional[Path] = None,
        image_stems: Optional[List[str]] = None,
        common_ids_only: bool = False,
        comparison_policy: Optional[str] = None,
    ) -> Dict:
        """Compare saved predictions with manual GT using strict or common-ID alignment."""
        predictions_dir = Path(predictions_dir).resolve()
        evaluation_output_dir = Path(evaluation_output_dir).resolve()
        override_dir = (
            Path(annotation_override_dir).resolve()
            if annotation_override_dir is not None
            else None
        )
        visualization_dir = evaluation_output_dir / 'visualizations'
        visualization_dir.mkdir(parents=True, exist_ok=True)

        if not predictions_dir.is_dir():
            raise FileNotFoundError(f"Prediction directory not found: {predictions_dir}")
        if not self.annotations_dir.is_dir():
            raise FileNotFoundError(f"Annotation directory not found: {self.annotations_dir}")

        annotation_suffix = '_shape.json'
        prediction_suffix = '_pred.json'
        base_annotation_stems = {
            path.name[:-len(annotation_suffix)]
            for path in self.annotations_dir.glob(f'*{annotation_suffix}')
        }
        override_annotation_stems = set()
        if override_dir is not None and override_dir.is_dir():
            override_annotation_stems = {
                path.name[:-len(annotation_suffix)]
                for path in override_dir.glob(f'*{annotation_suffix}')
            }
        annotation_stems = base_annotation_stems | override_annotation_stems
        prediction_stems = {
            path.name[:-len(prediction_suffix)]
            for path in predictions_dir.glob(f'*{prediction_suffix}')
        }
        if image_stems:
            selected_stems = {Path(stem).stem for stem in image_stems}
            annotation_stems &= selected_stems
            prediction_stems &= selected_stems
        annotation_only_images = sorted(annotation_stems - prediction_stems)
        prediction_only_images = sorted(prediction_stems - annotation_stems)
        if (
            not common_ids_only
            and (annotation_only_images or prediction_only_images)
        ):
            details = []
            if annotation_only_images:
                details.append(
                    'missing predictions: ' + ', '.join(annotation_only_images)
                )
            if prediction_only_images:
                details.append(
                    'predictions without GT: ' + ', '.join(prediction_only_images)
                )
            raise ValueError('Prediction/GT image sets differ; ' + '; '.join(details))
        evaluation_stems = annotation_stems & prediction_stems
        if not evaluation_stems:
            raise ValueError(f"No ground-truth annotations found in {self.annotations_dir}")

        all_annotations = []
        annotation_source_counts = Counter()
        population_counts = Counter()
        prediction_sam_config_counts = Counter()
        annotation_blinding_counts = Counter()
        alignment_transfer_counts = Counter()
        duplicate_annotation_rows_collapsed = 0
        for image_name in sorted(evaluation_stems):
            if image_name in override_annotation_stems:
                annotation = self.load_annotation(image_name, annotations_dir=override_dir)
                annotation_source_counts['override'] += 1
            else:
                annotation = self.load_annotation(image_name)
                annotation_source_counts['base'] += 1
            annotation_metadata = annotation.get('annotation_metadata', {}) if annotation else {}
            alignment_metadata = annotation_metadata.get('alignment', {})
            if isinstance(alignment_metadata, dict):
                for key in (
                    'new_mask_count',
                    'old_mask_count',
                    'hungarian_matched_pairs',
                    'transferred_valid_labels',
                    'new_unmatched_masks',
                    'old_unmatched_masks',
                    'source_annotation_gt_only_ids',
                ):
                    try:
                        alignment_transfer_counts[key] += int(
                            alignment_metadata.get(key, 0) or 0
                        )
                    except (TypeError, ValueError):
                        pass
            blinded_value = annotation_metadata.get('blinded_to_predictions')
            if isinstance(blinded_value, bool):
                annotation_blinding_counts[
                    'blinded' if blinded_value else 'not_blinded'
                ] += 1
            else:
                annotation_blinding_counts['not_recorded'] += 1

            prediction_path = predictions_dir / f'{image_name}_pred.json'
            with open(prediction_path, 'r', encoding='utf-8') as handle:
                prediction_payload = json.load(handle)
            cached_sam_config = prediction_payload.get('sam_config')
            if not isinstance(cached_sam_config, dict):
                prediction_sam_config_counts['missing'] += 1
            elif self._sam_config_matches(cached_sam_config):
                prediction_sam_config_counts['matching'] += 1
            else:
                prediction_sam_config_counts['mismatching'] += 1
            loaded_prediction = self.load_prediction(image_name, predictions_dir)
            if annotation is None or loaded_prediction is None:
                raise ValueError(f"Could not load aligned GT and prediction for {image_name}")

            masks, predictions = loaded_prediction
            prediction_by_id = {}
            for prediction in predictions:
                particle_id = int(prediction['particle_id'])
                if particle_id in prediction_by_id:
                    raise ValueError(
                        f"Duplicate prediction particle_id {particle_id} in {image_name}"
                    )
                prediction_by_id[particle_id] = prediction

            annotation_particles = annotation.get('particles', [])
            annotation_rows_by_id = {}
            for item in annotation_particles:
                particle_id = int(item['particle_id'])
                annotation_rows_by_id.setdefault(particle_id, []).append(item)

            known_ids = set(annotation_rows_by_id)
            prediction_ids = set(prediction_by_id)
            gt_only_ids = sorted(known_ids - prediction_ids)
            prediction_only_ids = sorted(prediction_ids - known_ids)
            valid_gt_only_ids = {
                particle_id
                for particle_id in gt_only_ids
                if any(
                    not bool(row.get('skipped', False))
                    for row in annotation_rows_by_id[particle_id]
                )
            }
            if (
                not common_ids_only
                and (valid_gt_only_ids or prediction_only_ids)
            ):
                raise ValueError(
                    f"Particle IDs differ for {image_name}; "
                    f"missing valid predictions={sorted(valid_gt_only_ids)}, "
                    f"predictions without GT={prediction_only_ids}"
                )
            if len(masks) != len(predictions):
                raise ValueError(
                    f"Mask/prediction count differs for {image_name}: "
                    f"{len(masks)} vs {len(predictions)}"
                )

            population_counts['gt_only_particle_ids'] += len(gt_only_ids)
            population_counts['gt_only_valid_particle_ids'] += len(valid_gt_only_ids)
            population_counts['gt_only_skipped_particle_ids'] += (
                len(gt_only_ids) - len(valid_gt_only_ids)
            )
            population_counts['prediction_only_particle_ids'] += len(
                prediction_only_ids
            )

            common_ids = known_ids & prediction_ids
            population_counts['common_particle_ids'] += len(common_ids)
            canonical_annotation_by_id = {}
            for particle_id in common_ids:
                rows = annotation_rows_by_id[particle_id]
                valid_rows = [
                    row for row in rows if not bool(row.get('skipped', False))
                ]
                skipped_rows = [
                    row for row in rows if bool(row.get('skipped', False))
                ]
                normalized_labels = {
                    _normalize_shape_label(
                        row.get('gt_shape'),
                        self.shape_labels,
                        is_prediction=False,
                    )
                    for row in valid_rows
                }
                if None in normalized_labels:
                    unsupported = [
                        row.get('gt_shape')
                        for row in valid_rows
                        if _normalize_shape_label(
                            row.get('gt_shape'),
                            self.shape_labels,
                            is_prediction=False,
                        )
                        is None
                    ]
                    raise ValueError(
                        f"Unsupported GT label for {image_name} particle "
                        f"{particle_id}: {unsupported}"
                    )
                if (valid_rows and skipped_rows) or len(normalized_labels) > 1:
                    raise ValueError(
                        f"Conflicting duplicate GT rows for {image_name} "
                        f"particle {particle_id}"
                    )
                if len(rows) > 1:
                    duplicate_annotation_rows_collapsed += len(rows) - 1
                canonical_annotation_by_id[particle_id] = (
                    valid_rows[-1] if valid_rows else skipped_rows[-1]
                )

            for prediction in predictions:
                particle_id = int(prediction['particle_id'])
                if particle_id not in canonical_annotation_by_id:
                    continue
                annotation_particle = canonical_annotation_by_id[particle_id]
                skipped = bool(annotation_particle.get('skipped', False))
                gt_shape = _normalize_shape_label(
                    annotation_particle.get('gt_shape'),
                    self.shape_labels,
                    is_prediction=False,
                )
                pred_shape = _normalize_shape_label(
                    prediction.get('pred_shape'),
                    self.shape_labels,
                    is_prediction=True,
                )
                confidence = float(prediction.get('pred_confidence', 0.0))
                if not np.isfinite(confidence):
                    raise ValueError(
                        f"Non-finite confidence for {image_name} particle {particle_id}"
                    )

                all_annotations.append(
                    {
                        'image_name': image_name,
                        'particle_id': particle_id,
                        'gt_shape': gt_shape,
                        'pred_shape': pred_shape,
                        'pred_confidence': confidence,
                        'correct': None if skipped else bool(gt_shape == pred_shape),
                        'border_touching': annotation_particle.get(
                            'border_touching', False
                        ),
                        'skipped': skipped,
                    }
                )
                population_counts[
                    'evaluated_skipped_particle_ids'
                    if skipped
                    else 'evaluated_valid_particle_ids'
                ] += 1

        population_counts['common_images'] = len(evaluation_stems)
        population_counts['annotation_only_images'] = len(annotation_only_images)
        population_counts['prediction_only_images'] = len(prediction_only_images)
        if common_ids_only:
            for image_name in annotation_only_images:
                if image_name in override_annotation_stems:
                    annotation = self.load_annotation(
                        image_name, annotations_dir=override_dir
                    )
                else:
                    annotation = self.load_annotation(image_name)
                if annotation is None:
                    continue
                rows_by_id = {}
                for row in annotation.get('particles', []):
                    rows_by_id.setdefault(int(row['particle_id']), []).append(row)
                population_counts['annotation_only_image_particle_ids'] += len(
                    rows_by_id
                )
                population_counts['annotation_only_image_valid_particle_ids'] += sum(
                    any(not bool(row.get('skipped', False)) for row in rows)
                    for rows in rows_by_id.values()
                )
            for image_name in prediction_only_images:
                loaded_prediction = self.load_prediction(image_name, predictions_dir)
                if loaded_prediction is not None:
                    population_counts['prediction_only_image_particle_ids'] += len(
                        loaded_prediction[1]
                    )

        metrics = self.calculate_metrics(all_annotations)
        if not metrics:
            raise ValueError('No valid annotated particles were available for evaluation')

        prediction_sam_config_complete = (
            prediction_sam_config_counts['matching'] == len(evaluation_stems)
        )
        population_alignment_complete = bool(
            not common_ids_only
            and not annotation_only_images
            and not prediction_only_images
            and population_counts['gt_only_valid_particle_ids'] == 0
            and population_counts['prediction_only_particle_ids'] == 0
        )
        if annotation_blinding_counts['blinded'] == len(evaluation_stems):
            annotation_blinding_status = 'blinded_to_predictions'
        elif annotation_blinding_counts['not_blinded'] == len(evaluation_stems):
            annotation_blinding_status = 'not_blinded_to_predictions'
        elif annotation_blinding_counts['not_recorded'] == len(evaluation_stems):
            annotation_blinding_status = 'not_recorded'
        else:
            annotation_blinding_status = 'mixed_or_partially_recorded'
        manuscript_values = self.build_manuscript_values(
            metrics=metrics,
            image_count=len(evaluation_stems),
            population_alignment_complete=population_alignment_complete,
            prediction_sam_config_complete=prediction_sam_config_complete,
            annotation_blinding_status=annotation_blinding_status,
        )
        if not prediction_sam_config_complete:
            print(
                "[WARN] Prediction SAM provenance is missing or mismatched; "
                "Manuscript Values will report Publication_Ready=False"
            )

        self.create_confusion_matrix_plot(
            metrics['confusion_matrix'], visualization_dir / 'confusion_matrix.png'
        )
        self.create_per_class_chart(
            metrics, visualization_dir / 'per_class_performance.png'
        )
        self.create_confidence_calibration_plot(
            metrics['confidences'],
            metrics['correct'],
            visualization_dir / 'confidence_calibration.png',
        )
        self.create_summary_dashboard(
            metrics, visualization_dir / 'summary_dashboard.png'
        )
        self.create_border_comparison_chart(
            metrics, visualization_dir / 'border_comparison.png'
        )
        self.create_confidence_calibration_chart(
            metrics, visualization_dir / 'confidence_filtering_analysis.png'
        )
        self.create_per_image_label_comparison(
            all_annotations,
            self.dataset_dir,
            predictions_dir=predictions_dir,
            visualization_dir=visualization_dir,
        )

        valid_annotations = [
            item for item in all_annotations if not item.get('skipped', False)
        ]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        excel_path = evaluation_output_dir / f'shape_validation_results_{timestamp}.xlsx'
        summary_data = {
            'Metric': [
                'Total Particles',
                'Overall Accuracy',
                'Mean Confidence',
                '--- Border Analysis ---',
                'Border-Cut Particles',
                'Border-Cut Accuracy',
                'Internal Particles',
                'Internal Accuracy',
            ],
            'Value': [
                metrics['total_particles'],
                f"{metrics['accuracy']:.2%}",
                f"{metrics['mean_confidence']:.2%}",
                '',
                metrics['n_border'],
                f"{metrics['border_accuracy']:.2%}",
                metrics['n_internal'],
                f"{metrics['internal_accuracy']:.2%}",
            ],
        }
        detailed_data = [
            {
                'image_name': item['image_name'],
                'particle_id': item['particle_id'],
                'gt_shape': item['gt_shape'],
                'pred_shape': item['pred_shape'],
                'pred_confidence': item['pred_confidence'],
                'correct': item['correct'],
                'border_touching': item.get('border_touching', False),
            }
            for item in valid_annotations
        ]
        per_class_data = [
            {
                'Shape': label,
                'Precision': metrics['precision_per_class'][label],
                'Recall': metrics['recall_per_class'][label],
                'F1-Score': metrics['f1_per_class'][label],
                'Support': metrics['support_per_class'][label],
            }
            for label in self.shape_labels
        ]
        confidence_filtered_data = []
        for threshold in sorted(metrics['confidence_filtered_metrics']):
            values = metrics['confidence_filtered_metrics'][threshold]
            confidence_filtered_data.append(
                {
                    'Confidence Threshold': f'{threshold:.0%}',
                    'Accuracy': f"{values['accuracy']:.2%}",
                    'Sample Count': values['count'],
                    'Sample Retention': f"{values['percentage']:.1%}",
                    'Improvement vs Overall': (
                        f"{values['accuracy'] - metrics['accuracy']:.2%}"
                    ),
                }
            )

        comparison_policy_label = comparison_policy or (
            'Common image and particle IDs only'
            if common_ids_only
            else 'Strict image and particle ID alignment'
        )
        population_data = [
            {
                'Item': 'Comparison_Policy',
                'Value': comparison_policy_label,
            },
            {'Item': 'Common_Images', 'Value': len(evaluation_stems)},
            {
                'Item': 'Annotation_Only_Images',
                'Value': len(annotation_only_images),
            },
            {
                'Item': 'Annotation_Only_Image_IDs',
                'Value': ';'.join(annotation_only_images),
            },
            {
                'Item': 'Prediction_Only_Images',
                'Value': len(prediction_only_images),
            },
            {
                'Item': 'Prediction_Only_Image_IDs',
                'Value': ';'.join(prediction_only_images),
            },
            {
                'Item': 'Common_Particle_IDs',
                'Value': population_counts['common_particle_ids'],
            },
            {
                'Item': 'Evaluated_Valid_Particle_IDs',
                'Value': population_counts['evaluated_valid_particle_ids'],
            },
            {
                'Item': 'Evaluated_Skipped_Particle_IDs',
                'Value': population_counts['evaluated_skipped_particle_ids'],
            },
            {
                'Item': 'GT_Only_Particle_IDs_In_Common_Images',
                'Value': population_counts['gt_only_particle_ids'],
            },
            {
                'Item': 'Prediction_Only_Particle_IDs_In_Common_Images',
                'Value': population_counts['prediction_only_particle_ids'],
            },
            {
                'Item': 'Annotation_Only_Image_Particle_IDs',
                'Value': population_counts['annotation_only_image_particle_ids'],
            },
            {
                'Item': 'Duplicate_Annotation_Rows_Collapsed',
                'Value': duplicate_annotation_rows_collapsed,
            },
            {
                'Item': 'Prediction_SAM_Config_Matching_Images',
                'Value': prediction_sam_config_counts['matching'],
            },
            {
                'Item': 'Prediction_SAM_Config_Missing_Images',
                'Value': prediction_sam_config_counts['missing'],
            },
            {
                'Item': 'Prediction_SAM_Config_Mismatching_Images',
                'Value': prediction_sam_config_counts['mismatching'],
            },
            {
                'Item': 'Annotation_Blinding_Status',
                'Value': annotation_blinding_status,
            },
            {
                'Item': 'Hungarian_Matched_Mask_Pairs',
                'Value': alignment_transfer_counts['hungarian_matched_pairs'],
            },
            {
                'Item': 'Hungarian_Transferred_Valid_Labels',
                'Value': alignment_transfer_counts['transferred_valid_labels'],
            },
            {
                'Item': 'Hungarian_New_Unmatched_Masks',
                'Value': alignment_transfer_counts['new_unmatched_masks'],
            },
            {
                'Item': 'Hungarian_Old_Unmatched_Masks',
                'Value': alignment_transfer_counts['old_unmatched_masks'],
            },
        ]

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            pd.DataFrame(detailed_data).to_excel(
                writer, sheet_name='Detailed Results', index=False
            )
            pd.DataFrame(per_class_data).to_excel(
                writer, sheet_name='Per-Class Metrics', index=False
            )
            pd.DataFrame(
                {
                    'Category': ['Border-Cut Particles', 'Internal Particles'],
                    'Count': [metrics['n_border'], metrics['n_internal']],
                    'Accuracy': [
                        f"{metrics['border_accuracy']:.2%}",
                        f"{metrics['internal_accuracy']:.2%}",
                    ],
                }
            ).to_excel(writer, sheet_name='Border Analysis', index=False)
            pd.DataFrame(confidence_filtered_data).to_excel(
                writer, sheet_name='Confidence Filtering', index=False
            )
            pd.DataFrame(
                metrics['confusion_matrix'],
                columns=self.shape_labels,
                index=self.shape_labels,
            ).to_excel(writer, sheet_name='Confusion Matrix')
            pd.DataFrame(population_data).to_excel(
                writer, sheet_name='Evaluation Population', index=False
            )
            manuscript_values.to_excel(
                writer, sheet_name='Manuscript Values', index=False
            )

        manifest = {
            'mode': 'evaluate_existing_predictions',
            'created_at': datetime.now().astimezone().isoformat(timespec='seconds'),
            'predictions_dir': str(predictions_dir),
            'annotations_dir': str(self.annotations_dir.resolve()),
            'evaluation_output_dir': str(evaluation_output_dir),
            'annotation_override_dir': str(override_dir) if override_dir else None,
            'annotation_source_image_counts': dict(annotation_source_counts),
            'dataset_dir': str(self.dataset_dir.resolve()),
            'shape_labels': list(self.shape_labels),
            'comparison_policy': comparison_policy_label,
            'alignment_transfer_counts': dict(alignment_transfer_counts),
            'image_count': len(evaluation_stems),
            'annotation_only_images': annotation_only_images,
            'prediction_only_images': prediction_only_images,
            'population_counts': dict(population_counts),
            'duplicate_annotation_rows_collapsed': (
                duplicate_annotation_rows_collapsed
            ),
            'valid_particle_count': len(valid_annotations),
            'skipped_particle_count': len(all_annotations) - len(valid_annotations),
            'accuracy': float(metrics['accuracy']),
            'macro_f1': float(metrics['macro_f1']),
            'weighted_f1': float(metrics['weighted_f1']),
            'mean_confidence': float(metrics['mean_confidence']),
            'prediction_sam_config_counts': dict(prediction_sam_config_counts),
            'annotation_blinding_status': annotation_blinding_status,
            'outputs': {
                'excel': excel_path.name,
                'visualizations': 'visualizations',
                'label_comparisons': 'visualizations/label_comparisons',
            },
        }
        manifest_path = evaluation_output_dir / 'evaluation_run.json'
        with open(manifest_path, 'w', encoding='utf-8') as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)

        print(f"[SUCCESS] Evaluated {len(evaluation_stems)} common images")
        print(f"          Valid particles: {len(valid_annotations)}")
        print(
            "          Skipped common particle IDs: "
            f"{population_counts['evaluated_skipped_particle_ids']}"
        )
        if common_ids_only:
            print(
                "          Excluded GT-only particle IDs: "
                f"{population_counts['gt_only_particle_ids']}"
            )
            print(
                "          Excluded prediction-only particle IDs: "
                f"{population_counts['prediction_only_particle_ids']}"
            )
        print(f"          Accuracy: {metrics['accuracy']:.2%}")
        print(f"          Excel: {excel_path}")
        print(f"          Visualizations: {visualization_dir}")
        return {
            'metrics': metrics,
            'excel_path': excel_path,
            'visualization_dir': visualization_dir,
            'manifest_path': manifest_path,
        }

    # =========================================================================
    # GUI ANNOTATION TOOL (matplotlib interactive)
    # =========================================================================

    def run_annotation_gui(
        self,
        image_name: str,
        image: np.ndarray,
        masks: List[Dict],
        predictions: List[Dict],
        existing_gt_by_id: Optional[Dict[int, str]] = None,
    ) -> List[Dict]:
        """
        Interactive GUI for shape annotation

        Args:
            image_name: Image file name
            image: BM3D+Noise2SR preprocessed image
            masks: SAM masks
            predictions: CLIP predictions

        Returns:
            List of annotations (particle_id, gt_shape, pred_shape, pred_confidence)
        """
        annotations = []
        current_particle_idx = [0]  # Mutable for closure
        gui_state = {'finished': False}  # Use dict for mutable state in closures

        def store_annotation(particle_idx: int, annotation: Dict):
            """Keep exactly one annotation for each prediction particle ID."""
            particle_id = int(predictions[particle_idx]['particle_id'])
            annotation = dict(annotation)
            annotation['particle_id'] = particle_id
            annotations[:] = [
                item
                for item in annotations
                if int(item['particle_id']) != particle_id
            ]
            annotations.append(annotation)

        def create_visualization():
            """Create the GUI layout"""
            fig = plt.figure(figsize=(16, 12))
            # ?대?吏 ?곸뿭?????ш쾶: [?곷떒, 以묐떒, ?섎떒] = [3, 3, 1.2]
            gs = GridSpec(3, 2, figure=fig, hspace=0.25, wspace=0.25,
                         height_ratios=[3, 3, 1.2],
                         top=0.95, bottom=0.05, left=0.05, right=0.95)

            # Left: Original + SAM mask overlay
            ax_original = fig.add_subplot(gs[0:2, 0])

            # Right: Cropped particle + CLIP prediction
            ax_particle = fig.add_subplot(gs[0:2, 1])

            # Bottom: Shape selection buttons (?ш린???ъ슜?섏? ?딆?留??좎?)
            ax_buttons = fig.add_subplot(gs[2, :])
            ax_buttons.axis('off')

            return fig, ax_original, ax_particle, ax_buttons

        # 踰꾪듉 李몄“瑜??좎??섍린 ?꾪븳 由ъ뒪??
        button_refs = []

        def update_display(particle_idx):
            """Update display for current particle"""
            # ?댁쟾 踰꾪듉 李몄“ ?댁젣
            button_refs.clear()

            fig, ax_original, ax_particle, ax_buttons = create_visualization()

            # Title
            fig.suptitle(f"{image_name} | Particle {particle_idx + 1}/{len(masks)}",
                        fontsize=14, fontweight='bold')

            # Left: Original with mask overlay
            if len(image.shape) == 2:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = image.copy()

            # Draw all masks in gray
            for i, mask in enumerate(masks):
                seg = mask['segmentation']
                contours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if i == particle_idx:
                    cv2.drawContours(img_rgb, contours, -1, (0, 255, 0), 3)  # Current: Green
                else:
                    cv2.drawContours(img_rgb, contours, -1, (128, 128, 128), 1)  # Others: Gray

            ax_original.imshow(img_rgb)
            ax_original.set_title("Original Image + SAM Masks (Current: Green)", fontsize=12)
            ax_original.axis('off')

            # Right: Cropped particle
            mask = masks[particle_idx]
            pred = predictions[particle_idx]

            # Extract particle
            seg = mask['segmentation']
            mask_image = np.zeros_like(image, dtype=np.uint8)
            mask_image[seg] = image[seg]

            # Crop and zoom
            pil_image = self.crop_and_zoom_particle(mask_image)
            ax_particle.imshow(pil_image)
            particle_id = int(pred['particle_id'])
            existing_gt = (
                existing_gt_by_id.get(particle_id)
                if existing_gt_by_id is not None
                else None
            )
            if existing_gt is not None:
                title = (
                    f"Existing GT: {existing_gt}\n"
                    f"Saved CLIP Prediction: {pred['pred_shape']} "
                    f"({pred['pred_confidence']:.2%})"
                )
            else:
                title = (
                    f"CLIP Prediction: {pred['pred_shape']} "
                    f"({pred['pred_confidence']:.2%})"
                )
            ax_particle.set_title(
                title, fontsize=12, color='blue', fontweight='bold'
            )
            ax_particle.axis('off')

            # Bottom: Shape selection buttons
            n_labels = len(self.shape_labels)
            cols = 5
            rows = (n_labels + cols - 1) // cols

            for i, label in enumerate(self.shape_labels):
                row = i // cols
                col = i % cols

                # Button position [left, bottom, width, height]
                # 踰꾪듉???붾㈃ ?섎떒??諛곗튂 (0.15遺???쒖옉)
                left = 0.05 + col * 0.18
                bottom = 0.15 + (rows - 1 - row) * 0.10  # ?꾨옒?먯꽌 ?꾨줈 ?볤린
                width = 0.16
                height = 0.08  # ?믪씠 異뺤냼

                ax_btn = plt.axes([left, bottom, width, height])

                # 紐⑤뱺 踰꾪듉 ?숈씪 ?됱긽 (移섑똿 諛⑹?)
                color = 'lightgray'

                btn = Button(ax_btn, f"[{i+1}] {label}", color=color)
                btn.on_clicked(lambda event, s=label: on_shape_selected(s))  # 蹂?섎챸 蹂寃쎌쑝濡??대줈? 臾몄젣 ?닿껐
                button_refs.append(btn)  # 李몄“ ?좎?

            # Navigation buttons
            ax_prev = plt.axes([0.05, 0.05, 0.15, 0.08])
            btn_prev = Button(ax_prev, '<< Previous')
            btn_prev.on_clicked(lambda event: on_prev())
            button_refs.append(btn_prev)

            # Skip object button (orange)
            ax_skip = plt.axes([0.35, 0.05, 0.15, 0.08])
            btn_skip = Button(ax_skip, 'Skip Object', color='orange')
            btn_skip.on_clicked(lambda event: on_skip_object())
            button_refs.append(btn_skip)

            ax_next = plt.axes([0.80, 0.05, 0.15, 0.08])
            btn_next = Button(ax_next, 'Next >>')
            btn_next.on_clicked(lambda event: on_next())
            button_refs.append(btn_next)

            # Save button
            if particle_idx == len(masks) - 1:
                ax_save = plt.axes([0.55, 0.05, 0.20, 0.08])
                btn_save = Button(ax_save, 'Save & Finish', color='lightblue')
                btn_save.on_clicked(lambda event: on_save())
                button_refs.append(btn_save)

            # ?쇰툝濡쒗궧 紐⑤뱶濡??쒖떆
            plt.draw()
            plt.pause(0.001)
            plt.show(block=False)

        def on_shape_selected(shape):
            """Handle shape selection"""
            particle_idx = current_particle_idx[0]
            pred = predictions[particle_idx]
            mask_dict = masks[particle_idx]

            # Check if particle touches border
            border_touching = self.is_border_touching(mask_dict['segmentation'])

            store_annotation(particle_idx, {
                'gt_shape': shape,
                'pred_shape': pred['pred_shape'],
                'pred_confidence': pred['pred_confidence'],
                'correct': (shape == pred['pred_shape']),
                'border_touching': border_touching
            })

            print(f"  Particle {particle_idx + 1}: GT={shape}, Pred={pred['pred_shape']} ({'OK' if shape == pred['pred_shape'] else 'X'})")

            # Move to next particle
            plt.close('all')  # 紐⑤뱺 figure ?リ린
            if particle_idx < len(masks) - 1:
                current_particle_idx[0] += 1
                update_display(current_particle_idx[0])
            else:
                print("\n  All particles annotated!")
                gui_state['finished'] = True

        def on_prev():
            """Go to previous particle"""
            if current_particle_idx[0] > 0:
                current_particle_idx[0] -= 1
                plt.close('all')
                update_display(current_particle_idx[0])
            else:
                print("  Already at first particle")

        def on_skip_object():
            """Skip current object (exclude from statistics and visualization)"""
            particle_idx = current_particle_idx[0]
            pred = predictions[particle_idx]

            # Mark as skipped (will be filtered out later)
            store_annotation(particle_idx, {
                'gt_shape': None,  # Mark as skipped
                'pred_shape': pred['pred_shape'],
                'pred_confidence': pred['pred_confidence'],
                'correct': None,
                'border_touching': None,
                'skipped': True  # Flag for filtering
            })

            print(f"  Particle {particle_idx + 1}: SKIPPED (excluded from statistics)")

            # Move to next particle
            plt.close('all')
            if particle_idx < len(masks) - 1:
                current_particle_idx[0] += 1
                update_display(current_particle_idx[0])
            else:
                print("\n  All particles processed!")
                gui_state['finished'] = True

        def on_next():
            """Go to next particle (skip annotation) - uses prediction as default"""
            particle_idx = current_particle_idx[0]
            
            # Use prediction as default for current particle
            pred = predictions[particle_idx]
            mask_dict = masks[particle_idx]
            border_touching = self.is_border_touching(mask_dict['segmentation'])

            store_annotation(particle_idx, {
                'gt_shape': pred['pred_shape'],  # Default to prediction
                'pred_shape': pred['pred_shape'],
                'pred_confidence': pred['pred_confidence'],
                'correct': True,
                'border_touching': border_touching
            })
            print(f"  Particle {particle_idx + 1}: Accepted prediction ({pred['pred_shape']})")

            # Move to next particle or finish
            plt.close('all')
            if particle_idx < len(masks) - 1:
                current_particle_idx[0] += 1
                update_display(current_particle_idx[0])
            else:
                print("\n  All particles annotated!")
                gui_state['finished'] = True

        def on_save():
            """Save and finish"""
            print(f"\n  Saving {len(annotations)} annotations...")
            gui_state['finished'] = True
            plt.close('all')

        # Start GUI
        update_display(0)

        # GUI ?좎? - 紐⑤뱺 particle??泥섎━???뚭퉴吏 ?湲?
        while not gui_state['finished']:
            try:
                # Process GUI events
                if plt.get_fignums():  # If any figure is open
                    plt.pause(0.1)
                else:
                    # No figure open - check if we're done or transitioning
                    if gui_state['finished']:
                        break
                    # Brief pause to allow new figure to be created
                    import time
                    time.sleep(0.05)
                    # Double check after pause
                    if not plt.get_fignums() and not gui_state['finished']:
                        # Still no figure and not finished - wait a bit more
                        time.sleep(0.1)
                        if not plt.get_fignums() and not gui_state['finished']:
                            # Something went wrong, but don't break - keep waiting
                            pass
            except KeyboardInterrupt:
                print("\n  Annotation interrupted by user")
                break
            except Exception as e:
                # Log but don't break on other exceptions
                print(f"  GUI Warning: {e}")
                if gui_state['finished']:
                    break
                continue

        # Ensure all figures are closed
        plt.close('all')

        prediction_order = {
            int(prediction['particle_id']): index
            for index, prediction in enumerate(predictions)
        }
        return sorted(
            annotations,
            key=lambda item: prediction_order[int(item['particle_id'])],
        )

    # =========================================================================
    # METRICS CALCULATION
    # =========================================================================

    def calculate_metrics(self, all_annotations: List[Dict]) -> Dict:
        """
        Calculate comprehensive validation metrics

        Args:
            all_annotations: All particle annotations from all images

        Returns:
            Dictionary of metrics
        """
        if len(all_annotations) == 0:
            return {}

        # Filter out skipped objects
        valid_annotations = [ann for ann in all_annotations if not ann.get('skipped', False)]
        skipped_count = len(all_annotations) - len(valid_annotations)

        if skipped_count > 0:
            print(f"\n  ?뱄툘 Filtered out {skipped_count} skipped objects from statistics")

        if len(valid_annotations) == 0:
            print("  ?좑툘 WARNING: All objects were skipped!")
            return {}

        # Use only valid annotations for metrics
        all_annotations = valid_annotations

        # Extract GT and predictions
        gt_labels = [ann['gt_shape'] for ann in all_annotations]
        pred_labels = [ann['pred_shape'] for ann in all_annotations]
        confidences = [ann['pred_confidence'] for ann in all_annotations]
        correct = [ann['correct'] for ann in all_annotations]

        # Overall accuracy
        accuracy = sum(correct) / len(correct)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            gt_labels, pred_labels, labels=self.shape_labels, zero_division=0
        )

        # Confusion matrix
        cm = confusion_matrix(gt_labels, pred_labels, labels=self.shape_labels)
        macro_precision = float(np.mean(precision))
        macro_recall = float(np.mean(recall))
        macro_f1 = float(np.mean(f1))
        support_total = int(np.sum(support))
        weighted_f1 = float(
            np.average(f1, weights=support) if support_total > 0 else np.nan
        )
        off_diagonal_cm = cm.copy()
        np.fill_diagonal(off_diagonal_cm, 0)
        major_confusion_count = int(off_diagonal_cm.max())
        if major_confusion_count > 0:
            major_indices = np.argwhere(
                off_diagonal_cm == major_confusion_count
            )[0]
            major_confusion_gt = self.shape_labels[int(major_indices[0])]
            major_confusion_pred = self.shape_labels[int(major_indices[1])]
        else:
            major_confusion_gt = ''
            major_confusion_pred = ''

        # Shape distribution
        gt_counts = Counter(gt_labels)
        pred_counts = Counter(pred_labels)

        # Border-cut vs internal analysis
        border_touching = [ann.get('border_touching', False) for ann in all_annotations]
        border_particles = [ann for ann, bt in zip(all_annotations, border_touching) if bt]
        internal_particles = [ann for ann, bt in zip(all_annotations, border_touching) if not bt]

        border_accuracy = sum([ann['correct'] for ann in border_particles]) / len(border_particles) if len(border_particles) > 0 else 0
        internal_accuracy = sum([ann['correct'] for ann in internal_particles]) / len(internal_particles) if len(internal_particles) > 0 else 0

        # ?넅 Confidence-based filtering analysis (multiple thresholds)
        confidence_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        confidence_filtered_metrics = {}

        for threshold in confidence_thresholds:
            high_conf_annotations = [ann for ann in all_annotations if ann['pred_confidence'] >= threshold]

            if len(high_conf_annotations) > 0:
                high_conf_accuracy = sum([ann['correct'] for ann in high_conf_annotations]) / len(high_conf_annotations)
                high_conf_count = len(high_conf_annotations)
                high_conf_percentage = high_conf_count / len(all_annotations)

                # Per-class metrics for high-confidence predictions
                high_conf_gt = [ann['gt_shape'] for ann in high_conf_annotations]
                high_conf_pred = [ann['pred_shape'] for ann in high_conf_annotations]
                high_conf_precision, high_conf_recall, high_conf_f1, _ = precision_recall_fscore_support(
                    high_conf_gt, high_conf_pred, labels=self.shape_labels, zero_division=0
                )

                confidence_filtered_metrics[threshold] = {
                    'accuracy': high_conf_accuracy,
                    'count': high_conf_count,
                    'percentage': high_conf_percentage,
                    'precision_per_class': dict(zip(self.shape_labels, high_conf_precision)),
                    'recall_per_class': dict(zip(self.shape_labels, high_conf_recall)),
                    'f1_per_class': dict(zip(self.shape_labels, high_conf_f1))
                }
            else:
                confidence_filtered_metrics[threshold] = {
                    'accuracy': 0,
                    'count': 0,
                    'percentage': 0,
                    'precision_per_class': {label: 0 for label in self.shape_labels},
                    'recall_per_class': {label: 0 for label in self.shape_labels},
                    'f1_per_class': {label: 0 for label in self.shape_labels}
                }

        # ?넅 Combined filtering: Internal + High-Confidence
        internal_high_conf_annotations = [
            ann for ann in internal_particles
            if ann['pred_confidence'] >= 0.7  # 70% threshold
        ]
        internal_high_conf_accuracy = (
            sum([ann['correct'] for ann in internal_high_conf_annotations]) / len(internal_high_conf_annotations)
            if len(internal_high_conf_annotations) > 0 else 0
        )

        metrics = {
            'total_particles': len(all_annotations),
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'balanced_accuracy': macro_recall,
            'major_confusion_gt': major_confusion_gt,
            'major_confusion_pred': major_confusion_pred,
            'major_confusion_count': major_confusion_count,
            'precision_per_class': dict(zip(self.shape_labels, precision)),
            'recall_per_class': dict(zip(self.shape_labels, recall)),
            'f1_per_class': dict(zip(self.shape_labels, f1)),
            'support_per_class': dict(zip(self.shape_labels, support)),
            'confusion_matrix': cm,
            'gt_distribution': dict(gt_counts),
            'pred_distribution': dict(pred_counts),
            'mean_confidence': np.mean(confidences),
            'confidences': confidences,
            'correct': correct,
            # Border-cut analysis
            'border_touching': border_touching,
            'n_border': len(border_particles),
            'n_internal': len(internal_particles),
            'border_accuracy': border_accuracy,
            'internal_accuracy': internal_accuracy,
            # ?넅 Confidence-based analysis
            'confidence_filtered_metrics': confidence_filtered_metrics,
            'n_internal_high_conf': len(internal_high_conf_annotations),
            'internal_high_conf_accuracy': internal_high_conf_accuracy
        }

        return metrics

    def build_manuscript_values(
        self,
        metrics: Dict,
        image_count: int,
        population_alignment_complete: bool,
        prediction_sam_config_complete: bool,
        annotation_blinding_status: str = 'not_recorded',
    ) -> pd.DataFrame:
        """Return raw numeric values needed for the shape-validation manuscript."""
        publication_ready = bool(
            self.expected_images is not None
            and image_count == self.expected_images
            and population_alignment_complete
            and prediction_sam_config_complete
            and annotation_blinding_status
            in {'blinded_to_predictions', 'not_blinded_to_predictions'}
            and metrics.get('total_particles', 0) > 0
        )

        def row(item, value, unit, definition):
            return {
                'Item': item,
                'Value': value,
                'Unit': unit,
                'Definition': definition,
            }

        rows = [
            row(
                'Publication_Ready', publication_ready, 'boolean',
                (
                    'Expected image count, strict population alignment, '
                    'prediction SAM provenance, and annotation blinding status '
                    'must all be recorded'
                ),
            ),
            row(
                'Expected_Image_Count', self.expected_images, 'images',
                'Value supplied with --expected-images',
            ),
            row(
                'Analyzed_Image_Count', image_count, 'images',
                'Images included in the aligned evaluation population',
            ),
            row(
                'Classifiable_Particle_Count', metrics['total_particles'],
                'particles', 'Non-skipped aligned particle masks',
            ),
            row(
                'Overall_Accuracy', metrics['accuracy'], 'proportion',
                'Correct predictions divided by classifiable particles',
            ),
            row(
                'Macro_F1', metrics['macro_f1'], 'proportion',
                'Unweighted arithmetic mean of class F1 scores',
            ),
            row(
                'Weighted_F1', metrics['weighted_f1'], 'proportion',
                'Class F1 scores weighted by GT support',
            ),
            row(
                'Balanced_Accuracy', metrics['balanced_accuracy'], 'proportion',
                'Unweighted arithmetic mean of class recalls',
            ),
            row(
                'Macro_Precision', metrics['macro_precision'], 'proportion',
                'Unweighted arithmetic mean of class precision',
            ),
            row(
                'Macro_Recall', metrics['macro_recall'], 'proportion',
                'Unweighted arithmetic mean of class recall',
            ),
            row(
                'Major_Confusion_GT_Label', metrics['major_confusion_gt'],
                'label', 'GT label in the largest off-diagonal confusion cell',
            ),
            row(
                'Major_Confusion_Predicted_Label',
                metrics['major_confusion_pred'], 'label',
                'Predicted label in the largest off-diagonal confusion cell',
            ),
            row(
                'Major_Confusion_Count', metrics['major_confusion_count'],
                'particles', 'Largest off-diagonal confusion-matrix cell',
            ),
            row(
                'Population_Alignment_Complete',
                bool(population_alignment_complete), 'boolean',
                'No valid GT or prediction rows were excluded by common-ID evaluation',
            ),
            row(
                'Prediction_SAM_Config_Complete',
                bool(prediction_sam_config_complete), 'boolean',
                'Every prediction JSON records the active SAM settings',
            ),
            row(
                'Annotation_Blinding_Status', annotation_blinding_status,
                'status', 'Blinding provenance recorded by the annotation workflow',
            ),
            row(
                'SAM_Predicted_IoU_Threshold', self.pred_iou_thresh,
                'proportion', 'Active SAM2 automatic-mask-generator setting',
            ),
            row(
                'SAM_Stability_Score_Threshold', self.stability_score_thresh,
                'proportion', 'Active SAM2 automatic-mask-generator setting',
            ),
        ]
        for label in self.shape_labels:
            safe_label = re.sub(r'[^A-Za-z0-9]+', '_', label).strip('_')
            rows.extend(
                [
                    row(
                        f'{safe_label}_Support',
                        int(metrics['support_per_class'][label]),
                        'particles', f'GT support for {label}',
                    ),
                    row(
                        f'{safe_label}_Precision',
                        metrics['precision_per_class'][label],
                        'proportion', f'One-vs-rest precision for {label}',
                    ),
                    row(
                        f'{safe_label}_Recall',
                        metrics['recall_per_class'][label],
                        'proportion', f'One-vs-rest recall for {label}',
                    ),
                    row(
                        f'{safe_label}_F1', metrics['f1_per_class'][label],
                        'proportion', f'One-vs-rest F1 for {label}',
                    ),
                ]
            )
        return pd.DataFrame(
            rows, columns=['Item', 'Value', 'Unit', 'Definition']
        )

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    def create_confusion_matrix_plot(self, cm: np.ndarray, save_path: Path):
        """Create confusion matrix heatmap"""
        plt.figure(figsize=(12, 10))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.shape_labels,
                   yticklabels=self.shape_labels,
                   cbar_kws={'label': 'Count'})

        plt.xlabel('Predicted Shape', fontsize=12, fontweight='bold')
        plt.ylabel('Ground Truth Shape', fontsize=12, fontweight='bold')
        plt.title('Shape Classification Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def create_per_class_chart(self, metrics: Dict, save_path: Path):
        """Create per-class performance bar chart"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        labels = self.shape_labels
        precision = [metrics['precision_per_class'][l] for l in labels]
        recall = [metrics['recall_per_class'][l] for l in labels]
        f1 = [metrics['f1_per_class'][l] for l in labels]

        x = np.arange(len(labels))
        width = 0.6

        axes[0].bar(x, precision, width, color='steelblue', alpha=0.7)
        axes[0].set_ylabel('Precision', fontsize=12)
        axes[0].set_title('Precision by Shape', fontsize=13, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=45, ha='right')
        axes[0].set_ylim(0, 1.1)
        axes[0].grid(True, alpha=0.3, axis='y')

        axes[1].bar(x, recall, width, color='orange', alpha=0.7)
        axes[1].set_ylabel('Recall', fontsize=12)
        axes[1].set_title('Recall by Shape', fontsize=13, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=45, ha='right')
        axes[1].set_ylim(0, 1.1)
        axes[1].grid(True, alpha=0.3, axis='y')

        axes[2].bar(x, f1, width, color='green', alpha=0.7)
        axes[2].set_ylabel('F1-Score', fontsize=12)
        axes[2].set_title('F1-Score by Shape', fontsize=13, fontweight='bold')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(labels, rotation=45, ha='right')
        axes[2].set_ylim(0, 1.1)
        axes[2].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def create_confidence_calibration_plot(self, confidences: List[float], correct: List[bool],
                                           save_path: Path):
        """Create confidence vs accuracy calibration plot"""
        # Bin confidences
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(confidences, bins)

        bin_acc = []
        bin_conf = []
        bin_counts = []

        for i in range(1, len(bins)):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_acc.append(np.mean([correct[j] for j in range(len(correct)) if mask[j]]))
                bin_conf.append(np.mean([confidences[j] for j in range(len(confidences)) if mask[j]]))
                bin_counts.append(np.sum(mask))

        plt.figure(figsize=(10, 8))

        # Scatter plot
        plt.scatter(bin_conf, bin_acc, s=[c*10 for c in bin_counts], alpha=0.6)

        # Perfect calibration line
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration', linewidth=2)

        plt.xlabel('Confidence', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Confidence Calibration (size = bin count)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def create_summary_dashboard(self, metrics: Dict, save_path: Path):
        """Create comprehensive summary dashboard"""
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.4)

        # Title
        fig.suptitle('Shape Validation Summary Dashboard', fontsize=16, fontweight='bold')

        # 1. Confusion Matrix
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.shape_labels,
                   yticklabels=self.shape_labels,
                   ax=ax1, cbar_kws={'label': 'Count'})
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Ground Truth')
        ax1.set_title('Confusion Matrix')

        # 2. GT Distribution (?듭씪???쒖꽌)
        ax2 = fig.add_subplot(gs[0, 2])
        gt_dist = metrics['gt_distribution']
        # self.shape_labels ?쒖꽌濡??뺣젹
        gt_values = [gt_dist.get(label, 0) for label in self.shape_labels]
        ax2.bar(range(len(self.shape_labels)), gt_values, color='steelblue', alpha=0.7)
        ax2.set_xticks(range(len(self.shape_labels)))
        ax2.set_xticklabels(self.shape_labels, rotation=45, ha='right', fontsize=8)
        ax2.set_title('GT Distribution')
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Pred Distribution (?듭씪???쒖꽌)
        ax3 = fig.add_subplot(gs[1, 2])
        pred_dist = metrics['pred_distribution']
        # self.shape_labels ?쒖꽌濡??뺣젹
        pred_values = [pred_dist.get(label, 0) for label in self.shape_labels]
        ax3.bar(range(len(self.shape_labels)), pred_values, color='orange', alpha=0.7)
        ax3.set_xticks(range(len(self.shape_labels)))
        ax3.set_xticklabels(self.shape_labels, rotation=45, ha='right', fontsize=8)
        ax3.set_title('Pred Distribution')
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Metrics Table
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')

        metrics_text = f"""
        Overall Metrics:
        ??Total Particles: {metrics['total_particles']}
        ??Accuracy: {metrics['accuracy']:.2%}
        ??Mean Confidence: {metrics['mean_confidence']:.2%}

        Per-Class Performance:
        ??Mean Precision: {np.mean(list(metrics['precision_per_class'].values())):.2%}
        ??Mean Recall: {np.mean(list(metrics['recall_per_class'].values())):.2%}
        ??Mean F1-Score: {np.mean(list(metrics['f1_per_class'].values())):.2%}
        """

        ax4.text(0.5, 0.5, metrics_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='center', horizontalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def create_border_comparison_chart(self, metrics: Dict, save_path: Path):
        """Create border-cut vs internal particles accuracy comparison"""
        fig, ax = plt.subplots(figsize=(10, 6))

        categories = ['Border-Cut\nParticles', 'Internal\nParticles']
        accuracies = [metrics['border_accuracy'], metrics['internal_accuracy']]
        counts = [metrics['n_border'], metrics['n_internal']]
        colors = ['#FF6B6B', '#4ECDC4']

        bars = ax.bar(categories, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

        # Add count and accuracy labels
        for i, (bar, acc, count) in enumerate(zip(bars, accuracies, counts)):
            height = bar.get_height()
            # Accuracy value
            ax.text(bar.get_x() + bar.get_width()/2, height/2,
                   f'{acc:.1%}',
                   ha='center', va='center', fontsize=14, fontweight='bold', color='white')
            # Count above bar
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                   f'n={count}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel('Classification Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Shape Classification Accuracy:\nBorder-Cut vs Internal Particles',
                    fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def create_confidence_calibration_chart(self, metrics: Dict, save_path: Path):
        """
        Create confidence calibration and filtering analysis chart

        Shows:
        1. Accuracy vs confidence threshold
        2. Sample retention rate vs confidence threshold
        3. Combined filtering (Internal + High-Confidence)
        """
        fig = plt.figure(figsize=(16, 6))
        gs = GridSpec(1, 3, figure=fig, wspace=0.3)

        # Chart 1: Accuracy vs Confidence Threshold
        ax1 = fig.add_subplot(gs[0, 0])
        thresholds = sorted(metrics['confidence_filtered_metrics'].keys())
        accuracies = [metrics['confidence_filtered_metrics'][t]['accuracy'] for t in thresholds]

        ax1.plot(thresholds, accuracies, marker='o', linewidth=2, markersize=8, color='#3498db')
        ax1.axhline(y=metrics['accuracy'], color='red', linestyle='--', linewidth=2, label=f'Overall: {metrics["accuracy"]:.1%}')
        ax1.set_xlabel('Confidence Threshold', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Classification Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title('Accuracy vs Confidence Threshold', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        ax1.set_ylim(0, 1.05)

        # Add value labels
        for thresh, acc in zip(thresholds, accuracies):
            ax1.text(thresh, acc + 0.02, f'{acc:.1%}', ha='center', va='bottom', fontsize=9)

        # Chart 2: Sample Retention Rate
        ax2 = fig.add_subplot(gs[0, 1])
        percentages = [metrics['confidence_filtered_metrics'][t]['percentage'] * 100 for t in thresholds]
        counts = [metrics['confidence_filtered_metrics'][t]['count'] for t in thresholds]

        bars = ax2.bar(thresholds, percentages, color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Confidence Threshold', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Sample Retention (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Sample Retention vs Confidence Threshold', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, 110)

        # Add count labels
        for bar, count, pct in zip(bars, counts, percentages):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{pct:.0f}%\n(n={count})', ha='center', va='bottom', fontsize=9)

        # Chart 3: Combined Filtering Comparison
        ax3 = fig.add_subplot(gs[0, 2])
        categories = ['All\nParticles', 'Internal\nOnly', 'High-Conf\n(??0%)', 'Internal +\nHigh-Conf']
        accuracies_combined = [
            metrics['accuracy'],
            metrics['internal_accuracy'],
            metrics['confidence_filtered_metrics'][0.7]['accuracy'],
            metrics['internal_high_conf_accuracy']
        ]
        counts_combined = [
            metrics['total_particles'],
            metrics['n_internal'],
            metrics['confidence_filtered_metrics'][0.7]['count'],
            metrics['n_internal_high_conf']
        ]
        colors = ['#95a5a6', '#3498db', '#e74c3c', '#27ae60']

        bars = ax3.bar(categories, accuracies_combined, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax3.set_ylabel('Classification Accuracy', fontsize=12, fontweight='bold')
        ax3.set_title('Combined Filtering Strategy Comparison', fontsize=13, fontweight='bold')
        ax3.set_ylim(0, 1.1)
        ax3.grid(True, alpha=0.3, axis='y')

        # Add accuracy and count labels
        for bar, acc, count in zip(bars, accuracies_combined, counts_combined):
            # Accuracy inside bar
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                    f'{acc:.1%}', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
            # Count above bar
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'n={count}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def create_per_image_label_comparison(
        self,
        all_annotations: List[Dict],
        image_dir: Path,
        predictions_dir: Optional[Path] = None,
        visualization_dir: Optional[Path] = None,
    ):
        """
        Create per-image label comparison visualizations

        Args:
            all_annotations: All annotations with image names
            image_dir: Directory containing original images
        """
        # Group by image
        from collections import defaultdict
        images_data = defaultdict(list)

        for ann in all_annotations:
            img_name = ann.get('image_name', 'unknown')
            images_data[img_name].append(ann)

        # Create visualization directory
        visualization_root = (
            Path(visualization_dir) if visualization_dir is not None else self.viz_dir
        )
        viz_dir = visualization_root / "label_comparisons"
        viz_dir.mkdir(exist_ok=True, parents=True)

        print(f"\n?뱤 Generating per-image label comparisons for {len(images_data)} images...")

        saved_count = 0
        for img_name, particles in images_data.items():
            # Load original image
            img_path = image_dir / f"{img_name}.jpg"
            if not img_path.exists():
                img_path = image_dir / f"{img_name}.png"
            if not img_path.exists():
                img_path = image_dir / f"{img_name}.bmp"

            if not img_path.exists():
                print(f"  WARNING: Image not found for {img_name}")
                continue

            # Load prediction file to get masks
            pred_data = self.load_prediction(img_name, predictions_dir)
            if pred_data is None:
                if predictions_dir is not None:
                    print(f"  WARNING: Prediction not found for {img_name} in {predictions_dir}")
                    continue
                # Auto-generate prediction using SAM2+CLIP
                print(f"  No prediction for {img_name}, running SAM2+CLIP...")
                try:
                    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    preprocessed = self.preprocess_image(image, img_name)
                    masks = self.run_sam2_inference(preprocessed)
                    if len(masks) == 0:
                        print(f"  WARNING: No particles detected in {img_name}")
                        continue
                    shapes, confidences = self.classify_shapes_with_clip(masks, preprocessed)
                    predictions = [
                        {'particle_id': i, 'pred_shape': s, 'pred_confidence': c}
                        for i, (s, c) in enumerate(zip(shapes, confidences))
                    ]
                    self.save_predictions(img_name, masks, predictions, preprocessed.shape[:2])
                    image = preprocessed
                    print(f"  Generated and saved prediction for {img_name} ({len(masks)} particles)")
                except Exception as e:
                    print(f"  ERROR: Failed to generate prediction for {img_name}: {e}")
                    continue
            else:
                masks, predictions = pred_data
                if predictions_dir is not None:
                    cache_path = self._bm3d_noise2sr_cache_path(img_name)
                    image = self._load_bm3d_noise2sr_image(cache_path)
                    if image is None:
                        print(
                            f"  WARNING: Cached BM3D+Noise2SR image not found for "
                            f"{img_name}: {cache_path}"
                        )
                        continue
                else:
                    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    image = self.preprocess_image(image, img_name)

            mask_by_particle_id = {
                int(prediction['particle_id']): mask
                for mask, prediction in zip(masks, predictions)
            }

            # Filter out skipped particles
            valid_particles = [p for p in particles if not p.get('skipped', False)]
            if len(valid_particles) == 0:
                print(f"  SKIPPED: All particles were skipped in {img_name}")
                continue

            # Use ALL shape labels for legend (not just unique ones from this image)
            # This ensures consistent legend across all images
            all_shape_labels = self.shape_labels

            # Create comparison figure with legend BELOW images (compact layout)
            fig = plt.figure(figsize=(20, 11))
            
            # Create gridspec: top row for images, bottom row for legend (compact)
            gs = fig.add_gridspec(2, 2, height_ratios=[10, 0.8], hspace=0.08)
            
            # Image axes (top row)
            axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
            
            # Legend axes (bottom row, spanning both columns)
            ax_legend = fig.add_subplot(gs[1, :])
            ax_legend.axis('off')

            # Prepare both images
            if len(image.shape) == 2:
                img_pred = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                img_gt = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                img_pred = image.copy()
                img_gt = image.copy()

            # Use fixed color map from class for ALL labels (consistent across all images)
            gt_color_map = {shape: self.shape_color_map[shape]['bgr'] for shape in all_shape_labels if shape in self.shape_color_map}

            correct_count = 0

            # Draw annotations and predictions (valid particles only)
            particles = valid_particles
            for p in particles:
                pid = p['particle_id']
                pred_shape = p['pred_shape']
                gt_shape = p['gt_shape']
                conf = p['pred_confidence']
                is_correct = p['correct']
                border = p.get('border_touching', False)

                if is_correct:
                    correct_count += 1

                # Get mask
                mask_dict = mask_by_particle_id.get(int(pid))
                if mask_dict is not None:
                    mask = mask_dict['segmentation']

                    # Find contours
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if len(contours) > 0:
                        # Calculate centroid
                        M = cv2.moments(contours[0])
                        if M["m00"] > 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                        else:
                            cx, cy = mask.shape[1]//2, mask.shape[0]//2

                        # LEFT: Draw GT annotation with GT color
                        gt_color = gt_color_map.get(gt_shape, (255, 255, 255))
                        cv2.drawContours(img_gt, contours, -1, gt_color, 2)

                        # RIGHT: Draw CLIP prediction
                        # Blue for correct, Red for incorrect
                        pred_contour_color = (255, 0, 0) if is_correct else (0, 0, 255)  # BGR: Blue or Red
                        cv2.drawContours(img_pred, contours, -1, pred_contour_color, 2)

                        # Only show text for incorrect predictions
                        if not is_correct:
                            # Small font, no background, red text
                            error_text = f"GT:{gt_shape} Pred:{pred_shape}"
                            cv2.putText(img_pred, error_text, (cx-20, cy), cv2.FONT_HERSHEY_SIMPLEX,
                                      0.35, (0, 0, 255), 1, cv2.LINE_AA)  # Red text

            # Create legend below images (compact, single row)
            # Build legend with colored boxes and black text for ALL shape labels
            legend_elements = []
            for shape in all_shape_labels:
                if shape in self.shape_color_map:
                    color_rgb = self.shape_color_map[shape]['rgb']
                    # Normalize RGB to 0-1 for matplotlib
                    color_norm = tuple(c/255 for c in color_rgb)
                    # Create colored patch
                    patch = plt.Rectangle((0, 0), 1, 1, facecolor=color_norm, edgecolor='black', linewidth=1)
                    legend_elements.append((patch, shape))
            
            # Draw legend horizontally in single row (compact)
            if legend_elements:
                n_items = len(legend_elements)
                x_spacing = 1.0 / (n_items + 1)
                
                for i, (patch, label) in enumerate(legend_elements):
                    x_pos = x_spacing * (i + 1)
                    
                    # Draw colored box (compact size)
                    box_width = 0.02
                    box_height = 0.4
                    color_rgb = self.shape_color_map[label]['rgb']
                    color_norm = tuple(c/255 for c in color_rgb)
                    rect = plt.Rectangle((x_pos - box_width/2, 0.3), box_width, box_height,
                                         facecolor=color_norm, edgecolor='black', linewidth=1,
                                         transform=ax_legend.transAxes)
                    ax_legend.add_patch(rect)
                    # Draw label text in black
                    ax_legend.text(x_pos + box_width/2 + 0.005, 0.5, label,
                                  transform=ax_legend.transAxes,
                                  fontsize=9, fontweight='bold', color='black',
                                  verticalalignment='center')

            # Display images
            axes[0].imshow(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB))
            axes[0].set_title(f'{img_name} - Manual Annotation (GT)', fontsize=14, fontweight='bold')
            axes[0].axis('off')

            axes[1].imshow(cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB))
            accuracy = correct_count / len(particles) if len(particles) > 0 else 0
            axes[1].set_title(f'{img_name} - CLIP Prediction ({correct_count}/{len(particles)} correct = {accuracy:.1%})',
                             fontsize=14, fontweight='bold', color='green' if accuracy >= 0.8 else 'red')
            axes[1].axis('off')

            plt.tight_layout()
            plt.savefig(viz_dir / f"{img_name}_label_comparison.png", dpi=150, bbox_inches='tight')
            plt.close()
            saved_count += 1

        print(f"??Saved {saved_count} label comparison images to: {viz_dir}")

    def _save_complete_analysis_data(self, all_annotations: List[Dict], metrics: Dict, excel_path: Path):
        """?꾩쟾??遺꾩꽍 ?곗씠?????(?щ텇?앹슜)

        Args:
            all_annotations: 紐⑤뱺 ?낆옄 annotation 由ъ뒪??
            metrics: 怨꾩궛??紐⑤뱺 硫뷀듃由?
            excel_path: Excel ?뚯씪 寃쎈줈
        """
        import pickle
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_data_path = self.output_dir / f"analysis_data_{timestamp}.pkl"

        # ??ν븷 ?곗씠??援ъ꽦
        complete_data = {
            # ?뚯씠?꾨씪???ㅼ젙
            'config': {
                'dataset_dir': str(self.dataset_dir),
                'output_dir': str(self.output_dir),
                'shape_labels': self.shape_labels,
                'shape_preset': self.shape_preset,
                'timestamp': timestamp
            },

            # ?듭떖 寃곌낵 ?곗씠??
            'all_annotations': all_annotations,  # 紐⑤뱺 ?낆옄 annotation (image, mask, gt, pred, confidence, correct, border_touching)

            # 硫뷀듃由??곗씠??
            'metrics': metrics,  # confusion_matrix, precision/recall/f1 per class, accuracy, border analysis

            # ?뚯깮 ?곗씠??(?ш뎄??媛?ν븯吏留??몄쓽瑜??꾪빐 ???
            'gt_labels': [ann['gt_shape'] for ann in all_annotations if not ann.get('skipped', False)],
            'pred_labels': [ann['pred_shape'] for ann in all_annotations if not ann.get('skipped', False)],
            'confidences': [ann['pred_confidence'] for ann in all_annotations if not ann.get('skipped', False)],
            'correct_flags': [ann['correct'] for ann in all_annotations if not ann.get('skipped', False)],
            'border_touching': [ann.get('border_touching', False) for ann in all_annotations if not ann.get('skipped', False)],

            # 硫뷀? ?뺣낫
            'n_total_particles': len(all_annotations),
            'n_valid_particles': len([ann for ann in all_annotations if not ann.get('skipped', False)]),
            'n_skipped': len([ann for ann in all_annotations if ann.get('skipped', False)]),
            'n_images': len(set([ann.get('image_name', ann.get('image', 'unknown')) for ann in all_annotations])),
            'excel_path': str(excel_path)
        }

        # ???
        with open(analysis_data_path, 'wb') as f:
            pickle.dump(complete_data, f)

        print(f"\n??Complete analysis data saved to: {analysis_data_path}")
        print(f"   - {complete_data['n_valid_particles']} valid particles")
        print(f"   - {complete_data['n_images']} images")
        print(f"   - Confusion matrix, per-class metrics, border analysis")
        print(f"   - Configuration: shape_preset={self.shape_preset}, labels={self.shape_labels}")
        print(f"\n   ?뱷 Use this file for re-analysis without re-running the pipeline:")
        print(f"      import pickle")
        print(f"      with open('{analysis_data_path}', 'rb') as f:")
        print(f"          data = pickle.load(f)")
        print(f"      all_annotations = data['all_annotations']")
        print(f"      metrics = data['metrics']")
        print(f"      confusion_matrix = metrics['confusion_matrix']")

    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================

    def run_full_pipeline(self, n_samples: int = 10, sample_strategy: str = 'random'):
        """
        Run full validation pipeline

        Args:
            n_samples: Number of images to sample
            sample_strategy: 'random' or future options

        Returns:
            Dictionary with results and paths
        """
        print("\n" + "="*80)
        print("SHAPE VALIDATION PIPELINE")
        print("="*80)

        # Step 1: Sample images
        print(f"\n[Step 1] Sampling {n_samples} images...")
        all_images = sorted(list(self.dataset_dir.glob("*.jpg")) +
                           list(self.dataset_dir.glob("*.png")) +
                           list(self.dataset_dir.glob("*.bmp")))

        if sample_strategy == 'random':
            import random
            random.seed(42)
            sampled_images = random.sample(all_images, min(n_samples, len(all_images)))
        else:
            sampled_images = all_images[:n_samples]

        print(f"   Selected {len(sampled_images)} images")

        # Step 2: Check existing annotations
        annotated, unannotated = self.check_existing_annotations(sampled_images)

        # Step 2: Run SAM2 + CLIP on all images (both annotated and unannotated)
        # For annotated images, we just load the prediction
        # For unannotated images, we run SAM2 + CLIP
        print(f"\n[Step 2] Processing {len(sampled_images)} images...")
        print(f"         Already annotated: {len(annotated)}")
        print(f"         Need annotation: {len(unannotated)}")

        if len(unannotated) > 0:
            print(f"         You can leave and come back after SAM2+CLIP completes!")
            print(f"         Manual annotation will start when you're ready.")

        # 紐⑤뱺 ?대?吏?????泥섎━ (annotated???ы븿?섏뿬 visualization???ъ슜)
        pending_annotations = []  # (img_path, image, masks, predictions) ???
        regenerated_prediction_stems = set()

        for img_path in tqdm(sampled_images, desc="Processing images"):
            try:
                # Load image
                image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                preprocessed = self.preprocess_image(image, img_path.stem)

                # Check if prediction already exists
                existing_pred = self.load_prediction(
                    img_path.stem,
                    require_active_sam_config=True,
                )

                if existing_pred is not None:
                    # Use existing prediction (skip SAM2 re-inference)
                    masks, predictions = existing_pred
                    print(f"  [LOADED] Using existing prediction for {img_path.name} ({len(masks)} particles)")

                else:
                    # Run SAM2 + CLIP (no existing prediction)
                    masks = self.run_sam2_inference(preprocessed)

                    if len(masks) == 0:
                        print(f"  WARNING: No particles detected in {img_path.name}")
                        continue

                    # CLIP classification
                    shapes, confidences = self.classify_shapes_with_clip(masks, preprocessed)

                    # Create prediction dictionaries
                    predictions = [
                        {
                            'particle_id': i,
                            'pred_shape': shape,
                            'pred_confidence': conf
                        }
                        for i, (shape, conf) in enumerate(zip(shapes, confidences))
                    ]

                    # Save predictions with encoded masks
                    self.save_predictions(
                        img_path.stem,
                        masks,
                        predictions,
                        preprocessed.shape[:2],
                        sam_config=self.sam_config,
                    )
                    regenerated_prediction_stems.add(img_path.stem)

                # Add to pending annotations (whether loaded or newly created)
                pending_annotations.append((img_path, preprocessed, masks, predictions))

            except Exception as e:
                print(f"\nERROR: Failed processing {img_path.name}: {e}")
                continue

        print(f"\n[SUCCESS] Processing completed! {len(pending_annotations)} images ready")

        # Filter only unannotated images for manual annotation
        # Use stem names for comparison to avoid Path object comparison issues
        unannotated_stems = set(img.stem for img in unannotated)
        # A shape label is attached to a particular saved SAM mask/particle_id.
        # If SAM was rerun under another threshold, the old annotation cannot be
        # silently paired with the new mask ordering and must be reviewed again.
        needs_annotation_stems = unannotated_stems | regenerated_prediction_stems
        
        # Debug: Print what we're filtering
        print(f"\n[DEBUG] Annotation-required stems: {needs_annotation_stems}")
        print(f"[DEBUG] Pending annotations stems: {[p[0].stem for p in pending_annotations]}")
        
        unannotated_pending = [(img_path, image, masks, predictions)
                              for img_path, image, masks, predictions in pending_annotations
                              if img_path.stem in needs_annotation_stems]
        
        print(f"[DEBUG] Filtered unannotated_pending: {len(unannotated_pending)} images")
        for p in unannotated_pending:
            print(f"        - {p[0].name}")

        # Check for unannotated images that are NOT in pending_annotations
        # These need prediction generation (SAM2 + CLIP)
        pending_stems = set(p[0].stem for p in pending_annotations)
        missing_stems = needs_annotation_stems - pending_stems
        
        if missing_stems:
            print(f"\n[WARNING] {len(missing_stems)} unannotated images missing from pending_annotations:")
            for stem in missing_stems:
                print(f"          - {stem}")
            print(f"          These images need SAM2 + CLIP processing...")
            
            # Process missing images
            for stem in missing_stems:
                # Find the original image path
                img_path = None
                for sampled_img in sampled_images:
                    if sampled_img.stem == stem:
                        img_path = sampled_img
                        break
                
                if img_path is None:
                    print(f"          ERROR: Could not find image path for {stem}")
                    continue
                
                try:
                    print(f"          Processing {img_path.name}...")
                    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    preprocessed = self.preprocess_image(image, img_path.stem)
                    
                    # Run SAM2 + CLIP
                    masks = self.run_sam2_inference(preprocessed)
                    
                    if len(masks) == 0:
                        print(f"          WARNING: No particles detected in {img_path.name}, skipping")
                        continue
                    
                    # CLIP classification
                    shapes, confidences = self.classify_shapes_with_clip(masks, preprocessed)
                    
                    # Create prediction dictionaries
                    predictions = [
                        {
                            'particle_id': i,
                            'pred_shape': shape,
                            'pred_confidence': conf
                        }
                        for i, (shape, conf) in enumerate(zip(shapes, confidences))
                    ]
                    
                    # Save predictions
                    self.save_predictions(
                        img_path.stem,
                        masks,
                        predictions,
                        preprocessed.shape[:2],
                        sam_config=self.sam_config,
                    )
                    
                    # Add to unannotated_pending
                    unannotated_pending.append((img_path, preprocessed, masks, predictions))
                    print(f"          ??Added {img_path.name} ({len(masks)} particles)")
                    
                except Exception as e:
                    print(f"          ERROR processing {stem}: {e}")
                    continue

        if len(unannotated_pending) > 0:
            print(f"\n{'='*80}")
            print(f"Now starting manual annotation phase...")
            print(f"You can classify each particle by clicking the shape buttons.")
            print(f"{'='*80}")

        # Step 3: Manual annotation (only for unannotated images)
        print(f"\n[Step 3] Manual annotation for {len(unannotated_pending)} images...")

        for idx, (img_path, image, masks, predictions) in enumerate(unannotated_pending):
            try:
                print(f"\n  [{idx+1}/{len(unannotated_pending)}] Annotating {img_path.name} ({len(masks)} particles)...")
                annotations = self.run_annotation_gui(img_path.name, image, masks, predictions)

                # Check if annotations were collected
                if len(annotations) == 0:
                    print(f"  ?좑툘 WARNING: No annotations collected for {img_path.name}!")
                    print(f"     This may indicate a GUI issue. Skipping this image.")
                    continue

                # Save annotation
                self.save_annotation(
                    img_path.stem,
                    annotations,
                    metadata={
                        'sam_config': dict(self.sam_config),
                        'prediction_file': f'{img_path.stem}_pred.json',
                        'blinded_to_predictions': False,
                    },
                )
                print(f"  ??Saved annotation for {img_path.name} ({len(annotations)} particles)")

            except Exception as e:
                print(f"\n[ERROR] Failed annotation on {img_path.name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\n[Step 3 Complete] Finished annotating {len(unannotated_pending)} images")

        # Step 4: Load all annotations
        print(f"\n[Step 4] Calculating metrics...")
        all_annotations = []
        annotation_blinding_counts = Counter()

        print(f"[DEBUG] Loading annotations for {len(sampled_images)} sampled images:")
        for img_path in sampled_images:
            ann = self.load_annotation(img_path.stem)
            if ann is not None:
                print(f"        ??{img_path.stem}: {len(ann['particles'])} particles")
                annotation_metadata = ann.get('annotation_metadata', {})
                blinded_value = annotation_metadata.get('blinded_to_predictions')
                if isinstance(blinded_value, bool):
                    annotation_blinding_counts[
                        'blinded' if blinded_value else 'not_blinded'
                    ] += 1
                else:
                    annotation_blinding_counts['not_recorded'] += 1
                # Add image_name to each particle for per-image visualization
                for particle in ann['particles']:
                    particle['image_name'] = img_path.stem
                all_annotations.extend(ann['particles'])
            else:
                print(f"        ??{img_path.stem}: No annotation found")

        if len(all_annotations) == 0:
            print("[ERROR] No annotations found!")
            return None

        # Calculate metrics
        metrics = self.calculate_metrics(all_annotations)
        analyzed_image_count = int(sum(annotation_blinding_counts.values()))
        if annotation_blinding_counts['blinded'] == analyzed_image_count:
            annotation_blinding_status = 'blinded_to_predictions'
        elif annotation_blinding_counts['not_blinded'] == analyzed_image_count:
            annotation_blinding_status = 'not_blinded_to_predictions'
        elif annotation_blinding_counts['not_recorded'] == analyzed_image_count:
            annotation_blinding_status = 'not_recorded'
        else:
            annotation_blinding_status = 'mixed_or_partially_recorded'
        manuscript_values = self.build_manuscript_values(
            metrics=metrics,
            image_count=analyzed_image_count,
            population_alignment_complete=True,
            prediction_sam_config_complete=True,
            annotation_blinding_status=annotation_blinding_status,
        )

        # Step 5: Create visualizations
        print(f"\n[Step 5] Creating visualizations...")
        self.create_confusion_matrix_plot(metrics['confusion_matrix'],
                                         self.viz_dir / "confusion_matrix.png")
        self.create_per_class_chart(metrics, self.viz_dir / "per_class_performance.png")
        self.create_confidence_calibration_plot(metrics['confidences'], metrics['correct'],
                                               self.viz_dir / "confidence_calibration.png")
        self.create_summary_dashboard(metrics, self.viz_dir / "summary_dashboard.png")

        # NEW: Border-cut comparison
        print(f"  Creating border-cut comparison...")
        self.create_border_comparison_chart(metrics, self.viz_dir / "border_comparison.png")

        # ?넅 NEW: Confidence calibration and filtering analysis
        print(f"  Creating confidence calibration chart...")
        self.create_confidence_calibration_chart(metrics, self.viz_dir / "confidence_filtering_analysis.png")

        # NEW: Per-image label comparison
        print(f"  Creating per-image label comparisons...")
        self.create_per_image_label_comparison(all_annotations, self.dataset_dir)

        # Step 6: Save Excel report
        print(f"\n?뮶 Step 5: Saving Excel report...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = self.results_dir / f"shape_validation_results_{timestamp}.xlsx"

        # Create DataFrames
        summary_data = {
            'Metric': ['Total Particles', 'Overall Accuracy', 'Mean Confidence',
                      '--- Border Analysis ---',
                      'Border-Cut Particles', 'Border-Cut Accuracy',
                      'Internal Particles', 'Internal Accuracy'],
            'Value': [metrics['total_particles'], f"{metrics['accuracy']:.2%}",
                     f"{metrics['mean_confidence']:.2%}",
                     '',  # Separator
                     metrics['n_border'], f"{metrics['border_accuracy']:.2%}",
                     metrics['n_internal'], f"{metrics['internal_accuracy']:.2%}"]
        }

        detailed_data = []
        for ann in all_annotations:
            detailed_data.append({
                'image_name': ann.get('image_name', ''),
                'particle_id': ann['particle_id'],
                'gt_shape': ann['gt_shape'],
                'pred_shape': ann['pred_shape'],
                'pred_confidence': ann['pred_confidence'],
                'correct': ann['correct'],
                'border_touching': ann.get('border_touching', False)
            })

        per_class_data = []
        for label in self.shape_labels:
            per_class_data.append({
                'Shape': label,
                'Precision': metrics['precision_per_class'][label],
                'Recall': metrics['recall_per_class'][label],
                'F1-Score': metrics['f1_per_class'][label],
                'Support': metrics['support_per_class'][label]
            })

        # Border comparison data
        border_comparison_data = {
            'Category': ['Border-Cut Particles', 'Internal Particles'],
            'Count': [metrics['n_border'], metrics['n_internal']],
            'Accuracy': [f"{metrics['border_accuracy']:.2%}", f"{metrics['internal_accuracy']:.2%}"]
        }

        # Save to Excel
        # ?넅 Confidence-filtered metrics data
        confidence_filtered_data = []
        for threshold in sorted(metrics['confidence_filtered_metrics'].keys()):
            conf_metrics = metrics['confidence_filtered_metrics'][threshold]
            confidence_filtered_data.append({
                'Confidence Threshold': f'{threshold:.0%}',
                'Accuracy': f"{conf_metrics['accuracy']:.2%}",
                'Sample Count': conf_metrics['count'],
                'Sample Retention': f"{conf_metrics['percentage']:.1%}",
                'Improvement vs Overall': f"{(conf_metrics['accuracy'] - metrics['accuracy']):.2%}"
            })

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            pd.DataFrame(detailed_data).to_excel(writer, sheet_name='Detailed Results', index=False)
            pd.DataFrame(per_class_data).to_excel(writer, sheet_name='Per-Class Metrics', index=False)
            pd.DataFrame(border_comparison_data).to_excel(writer, sheet_name='Border Analysis', index=False)
            # ?넅 Confidence filtering analysis
            pd.DataFrame(confidence_filtered_data).to_excel(writer, sheet_name='Confidence Filtering', index=False)

            # Confusion matrix
            cm_df = pd.DataFrame(metrics['confusion_matrix'],
                                columns=self.shape_labels,
                                index=self.shape_labels)
            cm_df.to_excel(writer, sheet_name='Confusion Matrix')
            manuscript_values.to_excel(
                writer, sheet_name='Manuscript Values', index=False
            )

        print(f"\n??Excel saved: {excel_path}")

        # ?넅 Save complete analysis data for re-analysis
        self._save_complete_analysis_data(
            all_annotations=all_annotations,
            metrics=metrics,
            excel_path=excel_path
        )

        # Final summary
        print("\n" + "="*80)
        print("??VALIDATION COMPLETE!")
        print("="*80)
        print(f"\n?뱤 Overall Results:")
        print(f"   Total Particles: {metrics['total_particles']}")
        print(f"   Accuracy: {metrics['accuracy']:.2%}")
        print(f"   Mean Confidence: {metrics['mean_confidence']:.2%}")
        print(f"\n?뵴 Border-Cut Analysis:")
        print(f"   Border-Cut Particles: {metrics['n_border']} (Accuracy: {metrics['border_accuracy']:.2%})")
        print(f"   Internal Particles: {metrics['n_internal']} (Accuracy: {metrics['internal_accuracy']:.2%})")
        print(f"\n?뱚 Output:")
        print(f"   Excel: {excel_path}")
        print(f"   Visualizations: {self.viz_dir}")
        print(f"   Label Comparisons: {self.viz_dir / 'label_comparisons'}")

        return {
            'metrics': metrics,
            'excel_path': excel_path,
            'viz_dir': self.viz_dir,
            'all_annotations': all_annotations
        }


# ============================================================================
# EXCEL REUSE + PUBLICATION FIGURE (1x2)
# ============================================================================

EVAL_CLASSES_2D = ["Circular", "Triangular", "Quadrilateral", "Hexagonal", "Irregular"]


def _normalize_shape_label(value, classes: List[str], is_prediction: bool) -> Optional[str]:
    """Normalize shape label to canonical class names."""
    class_lookup = {c.lower(): c for c in classes}

    if pd.isna(value):
        return "Irregular" if is_prediction else None

    text = str(value).strip()
    if not text:
        return "Irregular" if is_prediction else None

    text = text.replace("_", " ").replace("-", " ")
    text = " ".join(text.split())
    lower_text = text.lower()

    if lower_text in class_lookup:
        return class_lookup[lower_text]

    alias_groups = [
        {"circle", "circular"},
        {"triangle", "triangular"},
        {"quadrilateral"},
        {"hexagon", "hexagonal"},
        {"irregular"},
    ]
    for aliases in alias_groups:
        if lower_text not in aliases:
            continue
        for alias in aliases:
            if alias in class_lookup:
                return class_lookup[alias]

    compact = lower_text.replace(" ", "")
    for cls in classes:
        if compact == cls.lower().replace(" ", ""):
            return cls

    if is_prediction:
        return "Irregular"
    return None


def _resolve_shape_results_excel_path(results_excel_path: str) -> Path:
    """Resolve existing shape validation excel path (allows stem without .xlsx)."""
    if not results_excel_path:
        raise ValueError("results_excel_path is empty")

    raw = Path(results_excel_path)
    raw_with_xlsx = raw if raw.suffix.lower() == ".xlsx" else raw.with_suffix(".xlsx")

    validation_root = Path(str(get_validation_dir()))
    candidates = [raw, raw_with_xlsx]
    if not raw.is_absolute():
        candidates.extend(
            [
                Path.cwd() / raw,
                Path.cwd() / raw_with_xlsx,
                validation_root / "shape_validation" / "results" / raw,
                validation_root / "shape_validation" / "results" / raw_with_xlsx,
                validation_root / raw,
                validation_root / raw_with_xlsx,
            ]
        )

    seen = set()
    deduped = []
    for c in candidates:
        key = str(c)
        if key not in seen:
            deduped.append(c)
            seen.add(key)

    for candidate in deduped:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    attempted = "\n".join(f"  - {str(c)}" for c in deduped)
    raise FileNotFoundError(f"Could not find shape results excel.\nTried:\n{attempted}")


def _load_shape_eval_dataframe(excel_path: Path, classes: List[str]) -> pd.DataFrame:
    """
    Load and clean shape evaluation dataframe from excel.
    Uses ALL rows as test samples after GT label filtering.
    """
    xls = pd.ExcelFile(excel_path)
    target_sheet = "Detailed Results" if "Detailed Results" in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(excel_path, sheet_name=target_sheet)

    normalized_col_map = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_")
        if key in {"gt_shape", "ground_truth_shape", "ground_truth", "gt", "true_shape"}:
            normalized_col_map[col] = "gt_shape"
        elif key in {"pred_shape", "predicted_shape", "prediction", "pred"}:
            normalized_col_map[col] = "pred_shape"
        elif key in {"image_name", "image", "image_id", "img_name", "img"}:
            normalized_col_map[col] = "image_name"
        elif key in {"particle_id", "particle", "instance_id", "id"}:
            normalized_col_map[col] = "particle_id"
        elif key in {"pred_confidence", "confidence", "conf", "score"}:
            normalized_col_map[col] = "pred_confidence"
        elif key in {"correct", "is_correct"}:
            normalized_col_map[col] = "correct"
    if normalized_col_map:
        df = df.rename(columns=normalized_col_map)

    required = {"gt_shape", "pred_shape"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in excel: {missing}")

    if "image_name" not in df.columns:
        df["image_name"] = "unknown"
    if "particle_id" not in df.columns:
        df["particle_id"] = np.arange(len(df), dtype=int)

    df["gt_shape"] = df["gt_shape"].apply(lambda v: _normalize_shape_label(v, classes, is_prediction=False))
    df["pred_shape"] = df["pred_shape"].apply(lambda v: _normalize_shape_label(v, classes, is_prediction=True))

    # Keep only valid GT rows; map unknown/OOV predictions to Irregular by design.
    df = df[df["gt_shape"].isin(classes)].copy()
    df["pred_shape"] = df["pred_shape"].fillna("Irregular")

    # Ensure fixed category order for stable outputs
    df["gt_shape"] = pd.Categorical(df["gt_shape"], categories=classes, ordered=True)
    df["pred_shape"] = pd.Categorical(df["pred_shape"], categories=classes, ordered=True)
    df = df.reset_index(drop=True)

    return df


def generate_shape_publication_figure_from_excel(results_excel_path: str,
                                                 classes: Optional[List[str]] = None,
                                                 output_dir: Optional[str] = None) -> Dict[str, Path]:
    """
    Generate publication-quality 1x2 shape evaluation figure from existing excel.
    Saves:
      - shape_eval_main.pdf / shape_eval_main.png (600 dpi)
      - shape_eval_heatmap_only.pdf / shape_eval_heatmap_only.png (600 dpi)
    """
    if classes is None:
        classes = EVAL_CLASSES_2D

    excel_path = _resolve_shape_results_excel_path(results_excel_path)
    df = _load_shape_eval_dataframe(excel_path, classes)

    if len(df) == 0:
        raise ValueError("No valid rows after label cleaning/filtering.")

    y_true = df["gt_shape"].astype(str).tolist()
    y_pred = df["pred_shape"].astype(str).tolist()

    cm_counts = confusion_matrix(y_true, y_pred, labels=classes)
    overall_acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        labels=classes,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    macro_f1 = float(report["macro avg"]["f1-score"])
    weighted_f1 = float(report["weighted avg"]["f1-score"])

    row_sums = cm_counts.sum(axis=1, keepdims=True)
    cm_row_norm = np.divide(
        cm_counts * 100.0,
        row_sums,
        out=np.zeros_like(cm_counts, dtype=float),
        where=row_sums != 0,
    )

    per_class_rows = []
    for cls in classes:
        cls_rep = report.get(cls, {})
        per_class_rows.append(
            {
                "class": cls,
                "precision": float(cls_rep.get("precision", 0.0)),
                "recall": float(cls_rep.get("recall", 0.0)),
                "f1": float(cls_rep.get("f1-score", 0.0)),
                "support": int(cls_rep.get("support", 0)),
            }
        )
    per_class_df = pd.DataFrame(per_class_rows)

    aggregate_df = pd.DataFrame(
        [
            {"metric": "N images", "value": int(pd.Series(df["image_name"]).nunique())},
            {"metric": "N samples", "value": int(len(df))},
            {"metric": "Overall accuracy", "value": overall_acc},
            {"metric": "Balanced accuracy", "value": balanced_acc},
            {"metric": "Macro F1", "value": macro_f1},
            {"metric": "Weighted F1", "value": weighted_f1},
        ]
    )

    print("\n=== Per-class Metrics ===")
    print(per_class_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== Aggregate Metrics ===")
    print(aggregate_df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))

    cm_counts_df = pd.DataFrame(cm_counts, index=classes, columns=classes)
    cm_norm_df = pd.DataFrame(cm_row_norm, index=classes, columns=classes)
    print("\n=== Confusion Matrix (Counts) ===")
    print(cm_counts_df.to_string())
    print("\n=== Row-normalized Confusion Matrix (%) ===")
    print(cm_norm_df.to_string(float_format=lambda x: f"{x:.1f}"))

    out_dir = Path(output_dir).resolve() if output_dir else excel_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "shape_eval_main.pdf"
    png_path = out_dir / "shape_eval_main.png"
    notext_pdf_path = out_dir / "shape_eval_main_notext.pdf"
    notext_png_path = out_dir / "shape_eval_main_notext.png"
    heatmap_pdf_path = out_dir / "shape_eval_heatmap_only.pdf"
    heatmap_png_path = out_dir / "shape_eval_heatmap_only.png"

    sns.set_theme(style="white", context="paper")
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

    def _draw_figure(save_pdf: Path, save_png: Path, show_text: bool):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.6, 4.6), gridspec_kw={"wspace": 0.30})

        hm = sns.heatmap(
            cm_row_norm,
            ax=ax_a,
            cmap="Blues",
            vmin=0,
            vmax=100,
            annot=False,
            linewidths=0.6,
            linecolor="white",
            cbar=True,
            xticklabels=classes,
            yticklabels=classes,
        )
        cbar = hm.collections[0].colorbar
        if show_text:
            cbar.set_label("Row-normalized proportion (%)", fontsize=9)

            for i in range(len(classes)):
                for j in range(len(classes)):
                    pct = cm_row_norm[i, j]
                    count = int(cm_counts[i, j])
                    color = "white" if pct >= 55 else "black"
                    ax_a.text(
                        j + 0.5,
                        i + 0.5,
                        f"{pct:.1f}%\n(n={count})",
                        ha="center",
                        va="center",
                        fontsize=7.6,
                        color=color,
                    )

            ax_a.set_xlabel("Predicted Shape")
            ax_a.set_ylabel("Ground Truth Shape")
            ax_a.tick_params(axis="x", rotation=25)
            ax_a.tick_params(axis="y", rotation=0)
            ax_a.text(-0.12, 1.03, "(a)", transform=ax_a.transAxes, fontsize=10, fontweight="bold")
        else:
            ax_a.set_xlabel("")
            ax_a.set_ylabel("")
            ax_a.tick_params(axis="x", rotation=25, labelbottom=False, bottom=True)
            ax_a.tick_params(axis="y", rotation=0, labelleft=False, left=True)
            cbar.set_label("")
            cbar.ax.tick_params(labelleft=False, labelright=False, left=True, right=False)
            cbar.ax.set_yticklabels([])

        # Panel (b): Per-class metric bars (F1 + recall markers)
        x = np.arange(len(classes))
        f1_scores = per_class_df["f1"].to_numpy(dtype=float)
        recalls = per_class_df["recall"].to_numpy(dtype=float)
        bar_colors = sns.color_palette("colorblind", len(classes))

        ax_b.bar(x, f1_scores, color=bar_colors, alpha=0.92, edgecolor="black", linewidth=0.6)
        ax_b.scatter(x, recalls, color="black", s=18, zorder=3, label="Recall")

        if show_text:
            for idx, val in enumerate(f1_scores):
                ax_b.text(idx, min(1.0, val + 0.03), f"{val:.2f}", ha="center", va="bottom", fontsize=7.6)

            ax_b.set_xticks(x)
            ax_b.set_xticklabels(classes, rotation=25, ha="right")
            ax_b.set_ylim(0, 1.0)
            ax_b.set_ylabel("Score")
            ax_b.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.25)
            ax_b.text(-0.12, 1.03, "(b)", transform=ax_b.transAxes, fontsize=10, fontweight="bold")

            n_images = int(pd.Series(df["image_name"]).nunique()) if "image_name" in df.columns else 0
            stats_text = (
                f"N images = {n_images}\n"
                f"N samples = {len(df)}\n"
                f"Overall accuracy = {overall_acc:.3f}\n"
                f"Balanced accuracy = {balanced_acc:.3f}\n"
                f"Macro F1 = {macro_f1:.3f}\n"
                f"Weighted F1 = {weighted_f1:.3f}"
            )
            ax_b.text(
                0.98,
                0.03,
                stats_text,
                transform=ax_b.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.8,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.6", alpha=0.95),
            )
            ax_b.legend(loc="upper left", frameon=True)
        else:
            ax_b.set_xticks(x)
            ax_b.tick_params(axis="x", labelbottom=False, bottom=True)
            ax_b.tick_params(axis="y", labelleft=False, left=True)
            ax_b.set_ylim(0, 1.0)
            ax_b.set_ylabel("")
            ax_b.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.25)
            legend = ax_b.get_legend()
            if legend is not None:
                legend.remove()

        sns.despine(fig=fig)
        fig.tight_layout()
        fig.savefig(save_pdf, bbox_inches="tight")
        fig.savefig(save_png, dpi=600, bbox_inches="tight")
        plt.close(fig)

    def _draw_heatmap_only(save_pdf: Path, save_png: Path):
        """Draw only panel (a) heatmap with percentage labels (without count)."""
        fig, ax = plt.subplots(figsize=(5.7, 4.8))
        hm = sns.heatmap(
            cm_row_norm,
            ax=ax,
            cmap="Blues",
            vmin=0,
            vmax=100,
            annot=False,
            linewidths=0.6,
            linecolor="white",
            cbar=True,
            xticklabels=classes,
            yticklabels=classes,
        )
        cbar = hm.collections[0].colorbar
        cbar.set_label("Row-normalized proportion (%)", fontsize=9)

        for i in range(len(classes)):
            for j in range(len(classes)):
                pct = cm_row_norm[i, j]
                color = "white" if pct >= 55 else "black"
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{pct:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8.0,
                    color=color,
                )

        ax.set_xlabel("Predicted Shape")
        ax.set_ylabel("Ground Truth Shape")
        ax.tick_params(axis="x", rotation=25)
        ax.tick_params(axis="y", rotation=0)
        fig.tight_layout()
        fig.savefig(save_pdf, bbox_inches="tight")
        fig.savefig(save_png, dpi=600, bbox_inches="tight")
        plt.close(fig)

    _draw_figure(pdf_path, png_path, show_text=True)
    _draw_figure(notext_pdf_path, notext_png_path, show_text=False)
    _draw_heatmap_only(heatmap_pdf_path, heatmap_png_path)

    print(f"\nSaved figure PDF: {pdf_path}")
    print(f"Saved figure PNG: {png_path}")
    print(f"Saved no-text PDF: {notext_pdf_path}")
    print(f"Saved no-text PNG: {notext_png_path}")
    print(f"Saved heatmap-only PDF: {heatmap_pdf_path}")
    print(f"Saved heatmap-only PNG: {heatmap_png_path}")

    return {
        "excel_path": excel_path,
        "output_dir": out_dir,
        "pdf_path": pdf_path,
        "png_path": png_path,
        "notext_pdf_path": notext_pdf_path,
        "notext_png_path": notext_png_path,
        "heatmap_pdf_path": heatmap_pdf_path,
        "heatmap_png_path": heatmap_png_path,
        "metrics_per_class": per_class_df,
        "metrics_aggregate": aggregate_df,
        "cm_counts": cm_counts_df,
        "cm_row_normalized": cm_norm_df,
    }

# ============================================================================
# EXECUTION EXAMPLE
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shape validation pipeline / excel-based figure generation")
    parser.add_argument(
        "results_excel",
        nargs="?",
        default=None,
        help="Existing shape_validation_results_*.xlsx (or stem without .xlsx) for figure-only mode.",
    )
    parser.add_argument(
        "--results-excel",
        dest="results_excel_flag",
        type=str,
        default=None,
        help="Same as positional results_excel.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="Dataset_shape",
        help="Dataset directory for full pipeline mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(get_validation_dir()),
        help="Output root directory for full pipeline mode.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of sampled images in full pipeline mode.",
    )
    parser.add_argument(
        "--expected-images",
        type=int,
        default=None,
        help="Optional exact Dataset_shape image-count guard (use 100 for the full run).",
    )
    parser.add_argument(
        "--pred-iou-thresh",
        type=float,
        default=0.95,
        help="SAM2 predicted-IoU threshold selected by sam_param_optimizer.py.",
    )
    parser.add_argument(
        "--stability-score-thresh",
        type=float,
        default=0.80,
        help="SAM2 stability-score threshold selected by sam_param_optimizer.py.",
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help=(
            "Re-run only CLIP classification from cached BM3D+Noise2SR images "
            "and mask-bearing prediction JSON files."
        ),
    )
    parser.add_argument(
        "--reevaluate-prompts",
        action="store_true",
        help=(
            "Re-run CLIP with the current prompts, reuse annotation-compatible "
            "saved masks, annotate only SAM fallback images, and generate results."
        ),
    )
    parser.add_argument(
        "--fixed-setting-validation",
        action="store_true",
        help=(
            "Run/resume Shape validation with SAM2 at the requested thresholds, "
            "reuse pixel-compatible BM3D+Noise2SR/SAM caches, transfer existing "
            "manual labels by Hungarian mask-IoU matching, and evaluate without GUI."
        ),
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help=(
            "With --fixed-setting-validation, inspect cache reuse and required "
            "SAM2 work without loading SAM2/CLIP or writing predictions."
        ),
    )
    parser.add_argument(
        "--reuse-preprocessed-only",
        action="store_true",
        help=(
            "With --fixed-setting-validation, fail if a cached BM3D+Noise2SR "
            "image is unavailable; never run Noise2SR."
        ),
    )
    parser.add_argument(
        "--external-mask-cache-dir",
        type=Path,
        default=None,
        help=(
            "Optional fixed-threshold SAM2 *.pkl cache, normally the verified "
            "Size validation sam_p095_s080 cache."
        ),
    )
    parser.add_argument(
        "--external-preprocessed-dir",
        type=Path,
        default=None,
        help=(
            "Preprocessed images used to create --external-mask-cache-dir. "
            "Pixels must match exactly before a mask is reused."
        ),
    )
    parser.add_argument(
        "--alignment-iou-threshold",
        type=float,
        default=0.50,
        help="Minimum old-vs-new mask IoU for Hungarian GT-label transfer.",
    )
    parser.add_argument(
        "--review-mismatches",
        action="store_true",
        help=(
            "Open annotation GUI only for disagreements between the existing "
            "annotations and saved predictions; no model inference is run."
        ),
    )
    parser.add_argument(
        "--review-skipped",
        action="store_true",
        help=(
            "Open the annotation GUI for skipped objects whose original saved "
            "masks are available, then update the base annotations in place."
        ),
    )
    parser.add_argument(
        "--include-reviewed-skips",
        action="store_true",
        help=(
            "Compatibility option. --review-skipped now includes every currently "
            "skipped mask by default."
        ),
    )
    parser.add_argument(
        "--review-predictions-dir",
        type=Path,
        default=Path("predictions"),
        help=(
            "Saved prediction directory for --review-mismatches or --review-skipped. "
            "Relative paths are resolved under the "
            "shape_validation output directory."
        ),
    )
    parser.add_argument(
        "--review-output-name",
        type=Path,
        default=Path("annotations_mismatch_reviewed"),
        help=(
            "Reviewed annotation output for --review-mismatches. Relative paths "
            "are resolved under the shape_validation output directory."
        ),
    )
    parser.add_argument(
        "--prediction-output-name",
        type=str,
        default="predictions_2d_boundary_prompts",
        help=(
            "Output directory name under shape_validation for --predict-only "
            "or --reevaluate-prompts."
        ),
    )
    parser.add_argument(
        "--mask-source-dir",
        type=Path,
        action="append",
        default=None,
        help="Additional mask-bearing prediction directory; may be supplied multiple times.",
    )
    parser.add_argument(
        "--image-stem",
        action="append",
        default=None,
        help=(
            "Limit --predict-only, --reevaluate-prompts, --review-mismatches, "
            "or --review-skipped to one image stem; "
            "may be supplied multiple times."
        ),
    )
    parser.add_argument(
        "--annotation-override-name",
        type=str,
        default="annotations_2d_boundary_prompts",
        help=(
            "Directory name under shape_validation for annotations created from "
            "fallback SAM masks."
        ),
    )
    parser.add_argument(
        "--evaluate-predictions-dir",
        type=Path,
        default=None,
        help=(
            "Compare an existing *_pred.json directory with saved manual GT "
            "annotations without running preprocessing, SAM2, CLIP, or annotation GUI."
        ),
    )
    parser.add_argument(
        "--common-ids-only",
        action="store_true",
        help=(
            "With --evaluate-predictions-dir, evaluate only image/particle IDs "
            "present in both GT and predictions and report every exclusion."
        ),
    )
    parser.add_argument(
        "--evaluation-output-name",
        type=str,
        default=None,
        help=(
            "Output directory name under shape_validation. Defaults to results for "
            "existing-prediction evaluation and results_2d_boundary_prompts for "
            "--reevaluate-prompts."
        ),
    )
    args = parser.parse_args()

    for name, value in (
        ("--pred-iou-thresh", args.pred_iou_thresh),
        ("--stability-score-thresh", args.stability_score_thresh),
    ):
        if not 0.0 <= value <= 1.0:
            parser.error(f"{name} must be between 0 and 1")

    if args.common_ids_only and args.evaluate_predictions_dir is None:
        parser.error(
            "--common-ids-only requires --evaluate-predictions-dir"
        )
    if args.include_reviewed_skips and not args.review_skipped:
        parser.error("--include-reviewed-skips requires --review-skipped")
    if args.expected_images is not None and args.expected_images < 1:
        parser.error("--expected-images must be at least one")
    if args.plan_only and not args.fixed_setting_validation:
        parser.error("--plan-only requires --fixed-setting-validation")
    if args.reuse_preprocessed_only and not args.fixed_setting_validation:
        parser.error("--reuse-preprocessed-only requires --fixed-setting-validation")
    if not 0.0 <= args.alignment_iou_threshold <= 1.0:
        parser.error("--alignment-iou-threshold must be between 0 and 1")
    if (
        (args.external_mask_cache_dir is None)
        != (args.external_preprocessed_dir is None)
    ):
        parser.error(
            "--external-mask-cache-dir and --external-preprocessed-dir must be supplied together"
        )

    if args.expected_images is not None:
        dataset_path = Path(args.dataset_dir).resolve()
        if not dataset_path.is_dir():
            parser.error(f"Dataset directory not found: {dataset_path}")
        dataset_images = {
            path.resolve()
            for path in dataset_path.iterdir()
            if path.is_file()
            and path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}
        }
        if len(dataset_images) != args.expected_images:
            parser.error(
                f"Expected exactly {args.expected_images} Dataset_shape images, "
                f"found {len(dataset_images)} in {dataset_path}"
            )

    results_excel_input = args.results_excel_flag or args.results_excel
    selected_modes = sum(
        [
            bool(args.review_mismatches),
            bool(args.review_skipped),
            bool(args.reevaluate_prompts),
            bool(args.fixed_setting_validation),
            bool(args.predict_only),
            args.evaluate_predictions_dir is not None,
            bool(results_excel_input),
        ]
    )
    if selected_modes > 1:
        parser.error(
            "Choose only one of --review-mismatches, --review-skipped, "
            "--reevaluate-prompts, --fixed-setting-validation, --predict-only, "
            "--evaluate-predictions-dir, or results_excel"
        )

    def build_validator() -> ShapeValidator:
        return ShapeValidator(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            shape_preset='2D',
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_score_thresh,
            expected_images=args.expected_images,
        )

    if args.fixed_setting_validation:
        validator = build_validator()
        prediction_output = Path(args.prediction_output_name)
        if not prediction_output.is_absolute():
            prediction_output = validator.output_dir / prediction_output
        annotation_output = Path(args.annotation_override_name)
        if not annotation_output.is_absolute():
            annotation_output = validator.output_dir / annotation_output
        evaluation_output = Path(
            args.evaluation_output_name or "results_fixed_sam_setting"
        )
        if not evaluation_output.is_absolute():
            evaluation_output = validator.output_dir / evaluation_output
        validator.run_fixed_setting_validation(
            prediction_output_dir=prediction_output,
            annotation_output_dir=annotation_output,
            evaluation_output_dir=evaluation_output,
            external_mask_cache_dir=args.external_mask_cache_dir,
            external_preprocessed_dir=args.external_preprocessed_dir,
            image_stems=args.image_stem,
            reuse_preprocessed_only=args.reuse_preprocessed_only,
            alignment_iou_threshold=args.alignment_iou_threshold,
            plan_only=args.plan_only,
        )
        if args.plan_only:
            print("\nFixed-setting cache plan complete; no model inference was run.")
        else:
            print("\nFixed-setting SAM2 Shape validation complete!")
    elif args.review_skipped:
        validator = build_validator()
        review_predictions_dir = args.review_predictions_dir
        if not review_predictions_dir.is_absolute():
            review_predictions_dir = validator.output_dir / review_predictions_dir
        validator.review_skipped_annotations(
            predictions_dir=review_predictions_dir,
            image_stems=args.image_stem,
            additional_predictions_dirs=args.mask_source_dir,
        )
        print("\nSkipped-object annotation review complete!")
    elif args.review_mismatches:
        validator = build_validator()
        review_predictions_dir = args.review_predictions_dir
        if not review_predictions_dir.is_absolute():
            review_predictions_dir = validator.output_dir / review_predictions_dir
        review_output_dir = args.review_output_name
        if not review_output_dir.is_absolute():
            review_output_dir = validator.output_dir / review_output_dir
        validator.review_prediction_mismatches(
            predictions_dir=review_predictions_dir,
            annotations_dir=validator.annotations_dir,
            annotation_output_dir=review_output_dir,
            image_stems=args.image_stem,
        )
        print("\nSaved-prediction mismatch review complete!")
    elif args.reevaluate_prompts:
        validator = build_validator()
        prediction_output = Path(args.prediction_output_name)
        if not prediction_output.is_absolute():
            prediction_output = validator.output_dir / prediction_output
        annotation_override = Path(args.annotation_override_name)
        if not annotation_override.is_absolute():
            annotation_override = validator.output_dir / annotation_override
        evaluation_output = Path(
            args.evaluation_output_name or "results_2d_boundary_prompts"
        )
        if not evaluation_output.is_absolute():
            evaluation_output = validator.output_dir / evaluation_output
        validator.run_prompt_reevaluation(
            prediction_output_dir=prediction_output,
            annotation_override_dir=annotation_override,
            evaluation_output_dir=evaluation_output,
            additional_mask_source_dirs=args.mask_source_dir,
            image_stems=args.image_stem,
        )
        print("\nNew-prompt prediction and evaluation complete!")
    elif args.predict_only:
        validator = build_validator()
        prediction_output = Path(args.prediction_output_name)
        if not prediction_output.is_absolute():
            prediction_output = validator.output_dir / prediction_output
        validator.run_clip_prediction_only(
            prediction_output_dir=prediction_output,
            additional_mask_source_dirs=args.mask_source_dir,
            image_stems=args.image_stem,
        )
        print("\nCLIP-only prediction complete!")
    elif args.evaluate_predictions_dir is not None:
        validator = build_validator()
        evaluation_output = Path(args.evaluation_output_name or "results")
        if not evaluation_output.is_absolute():
            evaluation_output = validator.output_dir / evaluation_output
        annotation_override = Path(args.annotation_override_name)
        if not annotation_override.is_absolute():
            annotation_override = validator.output_dir / annotation_override
        validator.evaluate_prediction_directory(
            predictions_dir=args.evaluate_predictions_dir,
            evaluation_output_dir=evaluation_output,
            annotation_override_dir=(
                annotation_override if annotation_override.exists() else None
            ),
            image_stems=args.image_stem,
            common_ids_only=args.common_ids_only,
        )
        print("\nPrediction evaluation complete!")
    elif results_excel_input:
        generate_shape_publication_figure_from_excel(
            results_excel_path=results_excel_input,
            classes=EVAL_CLASSES_2D,
        )
    else:
        validator = build_validator()
        validator.run_full_pipeline(n_samples=args.n_samples)
        print("\nValidation complete!")
