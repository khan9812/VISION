"""
Size Validation Pipeline
- Annotation 蹂듭궗 諛?留ㅼ묶
- Template matching?쇰줈 crop ?곸뿭 李얘린
- Annotation ?섏젙 (醫뚰몴, bitmap)
- SAM2 ?덉륫
- GT-Pred 留ㅼ묶 諛??듦퀎 怨꾩궛
"""
import os
import sys
import argparse
# Add parent directory to path for modules access
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)


import os
import json
import cv2
import numpy as np
import base64
import zlib
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import shutil
from scipy.optimize import linear_sum_assignment
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import bm3d
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from modules.preprocessing import auto_preprocess
from modules.runtime_config import preprocessing_record, preprocessing_record_matches
from modules.project_paths import get_validation_dir, get_optimization_dir
from modules.sam2_utils import (
    filter_background_masks as filter_sam_background_masks,
    filter_overlapping_masks_by_centroid,
    get_sam2_config_path,
)

class SizeValidator:
    def __init__(self, dataset_dir: str, annotation_dir: str, output_dir: str,
                 filter_overlapping: bool = True,
                 use_noise2sr: bool = True,
                 noise2sr_epochs: Optional[int] = 1500,
                 pred_iou_thresh: float = 0.95,
                 stability_score_thresh: float = 0.80,
                 expected_images: Optional[int] = None,
                 reuse_preprocessed_only: bool = False):
        """
        Args:
            dataset_dir: Crop???대?吏 ?대뜑
            annotation_dir: ?먮낯 annotation ?대뜑
            output_dir: 寃곌낵 ????대뜑
            filter_overlapping: 寃뱀튂??留덉뒪???꾪꽣留??щ?
                - True: 寃뱀튂硫??묒? ?낆옄留??몄떇 (湲곗〈 諛⑹떇)
                - False: 寃뱀퀜??紐⑤몢 ?몄떇
            use_noise2sr: Noise2SR ?꾩쿂由??ъ슜 ?щ?
            noise2sr_epochs: Noise2SR ?숈뒿 epoch (None?대㈃ 湲곕낯媛?
        """
        self.dataset_dir = Path(dataset_dir)
        self.annotation_dir = Path(annotation_dir)

        # Size validation ?꾩슜 ?대뜑 ?앹꽦
        self.output_dir = Path(output_dir) / "size_validation"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Annotation ????대뜑
        self.matched_ann_dir = self.output_dir / "matched_annotations"
        self.matched_ann_dir.mkdir(exist_ok=True)

        # Preprocessing mode settings
        self.use_noise2sr = bool(use_noise2sr)
        self.noise2sr_epochs = noise2sr_epochs if noise2sr_epochs is not None else 1500
        self.pred_iou_thresh = float(pred_iou_thresh)
        self.stability_score_thresh = float(stability_score_thresh)
        self.expected_images = (
            int(expected_images) if expected_images is not None else None
        )
        self.reuse_preprocessed_only = bool(reuse_preprocessed_only)
        if not 0.0 <= self.pred_iou_thresh <= 1.0:
            raise ValueError("pred_iou_thresh must be between 0 and 1")
        if not 0.0 <= self.stability_score_thresh <= 1.0:
            raise ValueError("stability_score_thresh must be between 0 and 1")
        epochs_tag = f"e{self.noise2sr_epochs}" if self.noise2sr_epochs is not None else "edefault"
        if not self.use_noise2sr:
            raise ValueError("Size validation is fixed to BM3D + Noise2SR for this run.")
        # Do not mix the legacy partial cache (which could include BM3D fallback)
        # with this verified BM3D+Noise2SR validation run.
        self.preprocess_mode_tag = f"bm3d_noise2sr_{epochs_tag}_verified"
        self.sam_mode_tag = (
            f"sam_p{round(self.pred_iou_thresh * 100):03d}_"
            f"s{round(self.stability_score_thresh * 100):03d}"
        )
        self.last_preprocess_info = {}
        self.preprocessed_dir = self.output_dir / f"preprocessed_{self.preprocess_mode_tag}"
        self.preprocessed_dir.mkdir(exist_ok=True)
        self.compare_preprocess_dir = get_optimization_dir("preprocessing_comparison_full")
        self.optimizer_preprocess_dir = (
            get_optimization_dir("sam_optimization_v2")
            / "preprocessed_bm3d_noise2sr_e1500_verified"
        )

        # SAM2 segmentation cache ?대뜑 (overlap + preprocessing mode 蹂?遺꾨━)
        self.cache_dir = self.output_dir / (
            f"segmentation_cache_overlap_logic_{filter_overlapping}_"
            f"{self.preprocess_mode_tag}_{self.sam_mode_tag}"
        )
        self.cache_dir.mkdir(exist_ok=True)

        # SAM2 predictor (lazy loading)
        self._predictor = None

        # Overlap ?꾪꽣留??ㅼ젙
        self.filter_overlapping = filter_overlapping

        print(f"BM3D+Noise2SR preprocessing cache: {self.preprocessed_dir}")
        print(
            "SAM2 thresholds: "
            f"pred_iou={self.pred_iou_thresh:.2f}, "
            f"stability={self.stability_score_thresh:.2f}"
        )
        if self.reuse_preprocessed_only:
            print("Preprocessing policy: cache-only (fresh BM3D/Noise2SR is forbidden)")

    def _get_valid_segmentation_stats(self, required_cols: Optional[List[str]] = None,
                                      context: str = "") -> Optional[pd.DataFrame]:
        """Return segmentation_stats only when available and schema-compatible."""
        seg_stats = getattr(self, "segmentation_stats", None)
        if seg_stats is None:
            return None

        if not isinstance(seg_stats, pd.DataFrame):
            try:
                seg_stats = pd.DataFrame(seg_stats)
            except Exception:
                if context:
                    print(f"    WARNING: Invalid segmentation_stats type in {context}: {type(seg_stats)}")
                return None

        if seg_stats.empty:
            return None

        if required_cols:
            missing_cols = [col for col in required_cols if col not in seg_stats.columns]
            if missing_cols:
                suffix = f" in {context}" if context else ""
                print(f"    WARNING: segmentation_stats missing required columns{suffix}: {missing_cols}")
                return None

        return seg_stats

    def _get_visualization_dir(self) -> Path:
        """Visualization directory split by overlap logic + preprocessing mode."""
        return self.output_dir / (
            f"visualizations_overlap_logic_{self.filter_overlapping}_"
            f"{self.preprocess_mode_tag}_{self.sam_mode_tag}"
        )

    def _get_intermediate_results_path(self) -> Path:
        """Intermediate results path split by overlap logic + preprocessing mode."""
        return self.output_dir / (
            f"intermediate_results_overlap_logic_{self.filter_overlapping}_"
            f"{self.preprocess_mode_tag}_{self.sam_mode_tag}.pkl"
        )

    def load_sam2_model(self):
        """SAM2 紐⑤뜽 濡쒕뱶 (怨좎꽦???ㅼ젙: ?띾룄 臾댁떆, 理쒓퀬 ?덉쭏 ?곗꽑)"""
        if self._predictor is not None:
            return self._predictor

        print("Loading SAM2 model with HIGH PERFORMANCE settings...")

        # CUDA 泥댄겕
        if not torch.cuda.is_available():
            print("WARNING: CUDA is not available! Running on CPU will be very slow.")
            print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
            device = "cpu"
        else:
            device = "cuda"
            print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")

        project_root = Path(__file__).resolve().parent.parent
        sam2_checkpoint = project_root / "checkpoints" / "sam2.1_hiera_large.pt"
        model_cfg = Path(get_sam2_config_path("sam2.1_hiera_l.yaml"))
        if not sam2_checkpoint.is_file():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {sam2_checkpoint}")
        if not model_cfg.is_file():
            raise FileNotFoundError(f"SAM2 package config not found: {model_cfg}")

        sam2_model = build_sam2(str(model_cfg), str(sam2_checkpoint), device=device)
        # UNIFIED SAM2 PARAMETERS (consistent across all validation scripts)
        self._predictor = SAM2AutomaticMaskGenerator(
            sam2_model,
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
        predictor_module = type(self._predictor).__module__
        if not predictor_module.startswith("sam2."):
            raise RuntimeError(
                "Expected the SAM2 automatic mask generator, but loaded "
                f"{predictor_module}.{type(self._predictor).__name__}"
            )

        print(f"SAM2 model loaded on {device} with UNIFIED settings:")
        print(f"  - points_per_side=32, points_per_batch=256")
        print(
            f"  - pred_iou_thresh={self.pred_iou_thresh:.2f}, "
            f"stability_score_thresh={self.stability_score_thresh:.2f}"
        )
        print(f"  - crop_n_layers=1, use_m2m=True")
        print(f"  - filter_overlapping={self.filter_overlapping}")

        return self._predictor
    
    def load_image(self, image_path: str) -> np.ndarray:
        """?대?吏 濡쒕뱶"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return img
    
    def apply_bm3d_denoising(self, image: np.ndarray, sigma: float = 40) -> np.ndarray:
        """BM3D ?몄씠利??쒓굅"""
        # Normalize to [0, 1]
        img_normalized = image.astype(np.float32) / 255.0
        
        # Apply BM3D
        denoised = bm3d.bm3d(img_normalized, sigma_psd=sigma/255.0, stage_arg=bm3d.BM3DStages.ALL_STAGES)
        
        # Convert back to uint8
        denoised = np.clip(denoised * 255, 0, 255).astype(np.uint8)
        
        return denoised
    
    def _preprocessed_cache_path(self, image_name: str) -> Path:
        stem = Path(image_name).stem
        return self.preprocessed_dir / f"{stem}_2_bm3d_noise2sr_preprocessed.png"

    def _compare_preprocess_candidates(self, image_name: str) -> List[Path]:
        """Find verified reusable BM3D+Noise2SR outputs without running Noise2SR."""
        stem = Path(image_name).stem
        filename = f"{stem}_2_bm3d_noise2sr_preprocessed.png"
        direct_candidates = [
            self.optimizer_preprocess_dir / filename,
            self.compare_preprocess_dir / filename,
            self.compare_preprocess_dir / f"{stem}_preprocessed" / filename,
        ]
        if any(path.exists() for path in direct_candidates):
            return direct_candidates
        if self.compare_preprocess_dir.exists():
            return list(self.compare_preprocess_dir.rglob(filename))
        return []

    def verify_reusable_preprocessed_images(self, image_names) -> None:
        """Fail before SAM2 starts unless every image has a compatible cached stage."""
        allowed_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        dataset_by_stem = {
            path.stem: path
            for path in self.dataset_dir.iterdir()
            if path.is_file() and path.suffix.lower() in allowed_exts
        }
        missing = []
        incompatible = []
        for image_name in sorted(image_names):
            stem = Path(image_name).stem
            source_path = dataset_by_stem.get(stem)
            if source_path is None:
                missing.append(f"{stem} (source image missing)")
                continue
            source = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
            if source is None:
                missing.append(f"{stem} (source image unreadable)")
                continue
            candidates = [self._preprocessed_cache_path(stem), *self._compare_preprocess_candidates(stem)]
            existing = []
            compatible = False
            for candidate in dict.fromkeys(candidates):
                if not candidate.is_file():
                    continue
                existing.append(candidate)
                cached = cv2.imread(str(candidate), cv2.IMREAD_UNCHANGED)
                if cached is None:
                    continue
                cached = self._normalize_preprocessed_image(cached)
                if self._is_compatible_preprocessed_shape(source, cached):
                    compatible = True
                    break
            if not compatible:
                if existing:
                    incompatible.append(stem)
                else:
                    missing.append(stem)
        if missing or incompatible:
            details = []
            if missing:
                details.append("missing=" + ", ".join(missing[:20]))
            if incompatible:
                details.append("incompatible=" + ", ".join(incompatible[:20]))
            raise FileNotFoundError(
                "Cache-only size re-evaluation cannot start because reusable "
                "BM3D+Noise2SR stages are incomplete: " + "; ".join(details)
            )
        print(
            f"[OK] Cache-only preflight: compatible BM3D+Noise2SR stages "
            f"found for all {len(image_names)} images"
        )

    @staticmethod
    def _normalize_preprocessed_image(image: np.ndarray) -> np.ndarray:
        """Normalize cached images to BGR uint8 before converting for SAM."""
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

    @staticmethod
    def _is_compatible_preprocessed_shape(source: np.ndarray, preprocessed: np.ndarray) -> bool:
        """Noise2SR removes at most one trailing pixel from each odd dimension."""
        source_h, source_w = source.shape[:2]
        prep_h, prep_w = preprocessed.shape[:2]
        return (
            prep_h <= source_h
            and prep_w <= source_w
            and (source_h - prep_h) in (0, 1)
            and (source_w - prep_w) in (0, 1)
        )

    def _save_preprocessed_cache(self, image_name: str, image: np.ndarray, source: str) -> np.ndarray:
        preprocessed = self._normalize_preprocessed_image(image)
        cache_path = self._preprocessed_cache_path(image_name)
        if not cv2.imwrite(str(cache_path), preprocessed):
            raise IOError(f"Failed to save BM3D+Noise2SR image: {cache_path}")

        metadata = {
            "release_preprocessing": preprocessing_record(preprocessed.shape, self.noise2sr_epochs),
            "image_name": Path(image_name).stem,
            "method": "bm3d+noise2sr",
            "sigma_psd": 40,
            "noise2sr_epochs": self.noise2sr_epochs,
            "clahe_applied": False,
            "source": source,
            "shape": list(preprocessed.shape),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(cache_path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return preprocessed

    def preprocess_image(self, image: np.ndarray, image_name: Optional[str] = None) -> np.ndarray:
        """
        Run verified BM3D+Noise2SR preprocessing without CLAHE.

        Reuse order: size-validation cache, preprocessing-comparison output,
        then a fresh run that is saved for future executions. A BM3D-only
        fallback is intentionally not allowed.
        """
        if image is None:
            raise ValueError("Input image is None")

        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
        preprocessed_bgr = None

        if image_name:
            cache_path = self._preprocessed_cache_path(image_name)
            if cache_path.exists() and preprocessing_record_matches(cache_path, image.shape, self.noise2sr_epochs):
                cached = cv2.imread(str(cache_path), cv2.IMREAD_UNCHANGED)
                if cached is not None:
                    cached = self._normalize_preprocessed_image(cached)
                    if self._is_compatible_preprocessed_shape(image_bgr, cached):
                        print(f"   [CACHE] BM3D+Noise2SR: {cache_path.name}")
                        preprocessed_bgr = cached
                if preprocessed_bgr is None:
                    print(f"   [WARN] Ignoring incompatible preprocessing cache: {cache_path}")

            if preprocessed_bgr is None:
                for candidate in self._compare_preprocess_candidates(image_name):
                    if not candidate.exists() or not preprocessing_record_matches(candidate, image.shape, self.noise2sr_epochs):
                        continue
                    reused = cv2.imread(str(candidate), cv2.IMREAD_UNCHANGED)
                    if reused is None:
                        continue
                    reused = self._normalize_preprocessed_image(reused)
                    if not self._is_compatible_preprocessed_shape(image_bgr, reused):
                        print(f"   [WARN] Ignoring incompatible comparison output: {candidate}")
                        continue
                    print(f"   [REUSE] BM3D+Noise2SR: {candidate.name}")
                    preprocessed_bgr = self._save_preprocessed_cache(image_name, reused, str(candidate))
                    break

        if preprocessed_bgr is None:
            if self.reuse_preprocessed_only:
                raise FileNotFoundError(
                    "Cache-only size re-evaluation forbids running BM3D/Noise2SR; "
                    f"no compatible stage was found for {Path(image_name).stem if image_name else 'image'}"
                )
            default_preproc_config = str(
                Path(__file__).resolve().parent.parent / "configs" / "sam2.1" / "default_parameters.yaml"
            )
            preprocessed_bgr, preproc_info = auto_preprocess(
                image_bgr,
                config_path=default_preproc_config,
                sigma_psd=40,
                use_dataset_params=False,
                force_noise2sr=True,
                noise2sr_epochs=self.noise2sr_epochs,
            )
            if not (preproc_info.get("bm3d_applied") and preproc_info.get("noise2sr_applied")):
                raise RuntimeError(
                    "Expected BM3D+Noise2SR preprocessing, but "
                    f"got '{preproc_info.get('denoising_method', 'unknown')}'."
                )
            if preproc_info.get("clahe_applied"):
                raise RuntimeError("Size validation preprocessing must not include CLAHE.")
            if image_name:
                preprocessed_bgr = self._save_preprocessed_cache(
                    image_name,
                    preprocessed_bgr,
                    "generated_by_size_validation",
                )

        self.last_preprocess_info = {
            "denoising_method": "bm3d+noise2sr",
            "bm3d_applied": True,
            "noise2sr_applied": True,
            "clahe_applied": False,
        }
        return cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2GRAY)

    def filter_background_masks(self, masks: List[Dict], image_shape: Tuple[int, int],
                                border_tolerance: int = 5) -> List[Dict]:
        """諛곌꼍 留덉뒪???쒓굅 (4媛?寃쎄퀎 ?곗튂 OR 90% ?댁긽 硫댁쟻)

        Args:
            masks: SAM2 留덉뒪??由ъ뒪??
            image_shape: (height, width)
            border_tolerance: 寃쎄퀎 ?먯젙 ?덉슜 ?ㅼ감 (?쎌?)

        Returns:
            ?꾪꽣留곷맂 留덉뒪??由ъ뒪??
        """
        filtered_masks = filter_sam_background_masks(
            masks,
            image_shape,
            border_tolerance=border_tolerance,
        )
        removed_count = len(masks) - len(filtered_masks)
        if removed_count > 0:
            print(f"   Removed {removed_count} background masks (4-border touching OR >90% area)")
        return filtered_masks

    def run_sam2_inference(self, image: np.ndarray, image_name: str = None) -> List[np.ndarray]:
        """SAM2 異붾줎 ?ㅽ뻾 (罹먯떆 吏??

        Args:
            image: ?낅젰 ?대?吏
            image_name: ?대?吏 ?대쫫 (罹먯떆 ?ㅻ줈 ?ъ슜, ?뺤옣???쒖쇅)

        Returns:
            留덉뒪??由ъ뒪??
        """
        # 罹먯떆 ?뺤씤
        if image_name:
            cache_file = self.cache_dir / f"{image_name}_sam2_masks.pkl"
            if cache_file.exists():
                print(f"   Loading SAM2 masks from cache: {image_name}")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)

        # 罹먯떆媛 ?놁쑝硫?SAM2 異붾줎 ?ㅽ뻾
        predictor = self.load_sam2_model()

        # RGB濡?蹂??(SAM2??RGB ?낅젰 ?꾩슂)
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image

        # 留덉뒪???앹꽦
        masks = predictor.generate(image_rgb)

        # 1. 諛곌꼍 留덉뒪???쒓굅 (4媛?寃쎄퀎 ?곗튂 OR 90% ?댁긽 硫댁쟻)
        masks = self.filter_background_masks(masks, image.shape[:2])

        # 2. Overlap ?쒓굅: centroid containment ?꾪꽣留?(?듭뀡)
        if self.filter_overlapping:
            masks = self.filter_overlapping_masks(masks)

        # 留덉뒪?щ쭔 異붿텧
        mask_list = [m['segmentation'] for m in masks]

        # 罹먯떆 ???
        if image_name:
            with open(cache_file, 'wb') as f:
                pickle.dump(mask_list, f)
            print(f"   Saved SAM2 masks to cache: {image_name}")

        return mask_list

    def filter_invalid_gt_masks(self, gt_masks: List[np.ndarray],
                                crop_shape: Tuple[int, int],
                                border_margin: int = 5) -> Tuple[List[np.ndarray], List[bool]]:
        """GT 留덉뒪???꾪꽣留?諛?寃쎄퀎 ?곗튂 ?뚮옒洹?

        Args:
            gt_masks: Ground truth 留덉뒪??由ъ뒪??
            crop_shape: (height, width)
            border_margin: 寃쎄퀎 ?먯젙 留덉쭊 (?쎌?)

        Returns:
            (?좏슚??留덉뒪??由ъ뒪?? 寃쎄퀎 ?곗튂 ?뚮옒洹?由ъ뒪??
            寃쎄퀎 ?곗튂 ?뚮옒洹? True = crop 寃쎄퀎???우븘?덉쓬 (遺遺꾩쟻 object)
        """
        valid_masks = []
        boundary_flags = []

        for mask in gt_masks:
            # 留덉뒪?ш? 鍮꾩뼱?덉쑝硫??ㅽ궢
            coords = np.argwhere(mask > 0)
            if len(coords) == 0:
                continue

            # Crop ?대?吏 ?대????덈뒗 ?곸뿭留??대━??
            # Origin??諛뽰뿉 ?덉뼱???대?吏 ?대? ?곸뿭留??좏슚??mask濡??ъ슜
            mask_clipped = mask.copy()
            mask_clipped[:, :] = False  # 珥덇린??

            # Crop ?대? ?곸뿭留?True濡??ㅼ젙
            for y, x in coords:
                if 0 <= y < crop_shape[0] and 0 <= x < crop_shape[1]:
                    mask_clipped[y, x] = True

            # ?대━????留덉뒪?ш? 鍮꾩뼱?덉쑝硫??ㅽ궢
            clipped_coords = np.argwhere(mask_clipped > 0)
            if len(clipped_coords) == 0:
                continue

            y_min, x_min = clipped_coords.min(axis=0)
            y_max, x_max = clipped_coords.max(axis=0)

            # Crop 寃쎄퀎???우븘?덈뒗吏 ?뺤씤
            touches_boundary = (
                x_min <= border_margin or
                y_min <= border_margin or
                x_max >= (crop_shape[1] - border_margin) or
                y_max >= (crop_shape[0] - border_margin)
            )

            valid_masks.append(mask_clipped)
            boundary_flags.append(touches_boundary)

        return valid_masks, boundary_flags

    def align_gt_masks_to_observation_domain(
        self,
        gt_masks: List[np.ndarray],
        target_shape: Tuple[int, int],
        border_margin: int = 5,
    ) -> Tuple[List[np.ndarray], List[bool]]:
        """Crop GT masks to the exact BM3D+Noise2SR/SAM observation domain.

        Noise2SR can remove the final row and/or column from odd-sized inputs.
        Only that bottom/right one-pixel difference is accepted here. Larger
        differences are treated as a genuine alignment error.
        """
        target_h, target_w = map(int, target_shape)
        if target_h <= 0 or target_w <= 0:
            raise ValueError(f"Invalid comparison shape: {target_shape}")

        valid_masks: List[np.ndarray] = []
        boundary_flags: List[bool] = []
        for mask in gt_masks:
            mask_array = np.asarray(mask)
            if mask_array.ndim != 2:
                raise ValueError(f"GT mask must be 2D, got shape {mask_array.shape}")
            source_h, source_w = mask_array.shape
            delta_h = source_h - target_h
            delta_w = source_w - target_w
            if delta_h not in (0, 1) or delta_w not in (0, 1):
                raise ValueError(
                    "GT/preprocessed shape mismatch exceeds the supported "
                    f"bottom/right one-pixel crop: GT={mask_array.shape}, "
                    f"preprocessed={(target_h, target_w)}"
                )

            aligned = np.ascontiguousarray(mask_array[:target_h, :target_w] > 0)
            coords = np.argwhere(aligned)
            if len(coords) == 0:
                continue
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            touches_boundary = (
                x_min <= border_margin
                or y_min <= border_margin
                or x_max >= (target_w - border_margin)
                or y_max >= (target_h - border_margin)
            )
            valid_masks.append(aligned)
            boundary_flags.append(bool(touches_boundary))

        return valid_masks, boundary_flags

    def filter_overlapping_masks(self, masks: List[Dict]) -> List[Dict]:
        """Centroid containment 湲곕컲 overlap ?쒓굅"""
        return filter_overlapping_masks_by_centroid(masks)

    def step1_copy_matching_annotations(self):
        """Step 1: Dataset ?대?吏? 留ㅼ묶?섎뒗 annotation 蹂듭궗"""
        print("\n=== Step 1: Copying matching annotations ===")
        
        # Dataset ?대뜑??紐⑤뱺 ?대?吏 ?뚯씪紐?(?뺤옣???쒖쇅)
        image_files = list(self.dataset_dir.glob("*.png")) + \
                     list(self.dataset_dir.glob("*.jpg")) + \
                     list(self.dataset_dir.glob("*.bmp"))
        
        image_names = {f.stem for f in image_files}
        print(f"Found {len(image_names)} images in Dataset folder")
        
        # Annotation ?뚯씪 蹂듭궗
        copied_count = 0
        for img_name in image_names:
            ann_file = self.annotation_dir / f"{img_name}.png.json"
            if ann_file.exists():
                dest_file = self.matched_ann_dir / f"{img_name}.json"
                shutil.copy2(ann_file, dest_file)
                copied_count += 1
            else:
                print(f"Warning: No annotation found for {img_name}")
        
        print(f"Copied {copied_count} annotation files")
        return image_names
    
    def find_crop_offset(self, original_img: np.ndarray, cropped_img: np.ndarray) -> Tuple[int, int, float]:
        """Template matching?쇰줈 crop offset 李얘린 (confidence ?ы븿)"""
        result = cv2.matchTemplate(original_img, cropped_img, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Top-left corner of the match
        offset_x, offset_y = max_loc

        # Confidence check
        if max_val < 0.8:
            print(f"Warning: Template matching confidence is low: {max_val:.3f}")

        return offset_x, offset_y, max_val
    
    def decode_bitmap(self, bitmap_data: str) -> np.ndarray:
        """Decode Base64 PNG bitmap into a binary mask."""
        # Remove data URL prefix if exists
        if "base64," in bitmap_data:
            bitmap_data = bitmap_data.split("base64,")[1]
        
        # Remove all whitespace and newlines from base64 string
        bitmap_data = bitmap_data.replace('\n', '').replace('\r', '').replace(' ', '').strip()
        
        # Decode base64
        decoded = base64.b64decode(bitmap_data)
        
        # Check if data is zlib compressed (starts with 0x78 0x9C or similar)
        if len(decoded) > 2 and decoded[0] == 0x78:
            # Decompress zlib data
            decoded = zlib.decompress(decoded)
        
        # Convert to numpy array
        nparr = np.frombuffer(decoded, np.uint8)
        
        # Decode image
        mask = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            raise ValueError("Failed to decode bitmap - cv2.imdecode returned None")
        
        return mask
    
    def encode_bitmap(self, mask: np.ndarray) -> str:
        """Encode binary mask to Base64 PNG."""
        # Encode to PNG
        _, buffer = cv2.imencode('.png', mask)
        
        # Convert to base64
        encoded = base64.b64encode(buffer).decode('utf-8')
        
        # Add data URL prefix
        return f"data:image/png;base64,{encoded}"
    
    def step2_update_annotations(self, image_names: set, original_dir: Path):
        """Step 2: Template matching?쇰줈 annotation ?낅뜲?댄듃"""
        print("\n=== Step 2: Updating annotations with crop offsets ===")

        original_dir = Path(original_dir)
        if not original_dir.exists() or not original_dir.is_dir():
            raise FileNotFoundError(
                f"Original image directory not found: {original_dir}\n"
                f"Please set a valid path via --original-images-dir."
            )

        updated_annotations = {}
        self.low_confidence_images = []  # Template matching ??? confidence ?대?吏 異붿쟻
        missing_original_images = []

        # Build a stem -> original image path map once (case-insensitive, multiple formats).
        allowed_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        original_image_map = {}
        for path in original_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in allowed_exts:
                continue
            stem_key = path.stem.lower()
            if stem_key not in original_image_map:
                original_image_map[stem_key] = path

        for img_name in tqdm(image_names, desc="Processing annotations"):
            # ?먮낯 ?대?吏 濡쒕뱶
            original_img_path = original_image_map.get(img_name.lower())
            if original_img_path is None:
                missing_original_images.append(img_name)
                continue
            
            # Crop???대?吏 濡쒕뱶
            cropped_img_files = list(self.dataset_dir.glob(f"{img_name}.*"))
            if not cropped_img_files:
                continue
            
            original_img = cv2.imread(str(original_img_path), cv2.IMREAD_GRAYSCALE)
            cropped_img = cv2.imread(str(cropped_img_files[0]), cv2.IMREAD_GRAYSCALE)

            # Template matching
            offset_x, offset_y, confidence = self.find_crop_offset(original_img, cropped_img)
            crop_height, crop_width = cropped_img.shape

            # ??? confidence ?대?吏 湲곕줉
            if confidence < 0.8:
                self.low_confidence_images.append((img_name, confidence))
            
            # Annotation 濡쒕뱶 諛??섏젙
            ann_file = self.matched_ann_dir / f"{img_name}.json"
            with open(ann_file, 'r') as f:
                ann_data = json.load(f)
            
            # ?대?吏 ?ш린 ?낅뜲?댄듃
            ann_data['size']['height'] = crop_height
            ann_data['size']['width'] = crop_width
            
            # 媛?object ?낅뜲?댄듃
            updated_objects = []
            for obj in ann_data.get('objects', []):
                # Origin 醫뚰몴 蹂??
                orig_x, orig_y = obj['bitmap']['origin']
                new_orig_x = orig_x - offset_x
                new_orig_y = orig_y - offset_y
                
                # Bitmap ?붿퐫??
                bitmap_mask = self.decode_bitmap(obj['bitmap']['data'])
                
                # Crop ?곸뿭 ?댁뿉 ?덈뒗吏 ?뺤씤
                bitmap_h, bitmap_w = bitmap_mask.shape
                
                # ??origin??crop ?곸뿭 ?댁뿉 ?덈뒗吏 ?뺤씤
                if (new_orig_x + bitmap_w < 0 or new_orig_x >= crop_width or
                    new_orig_y + bitmap_h < 0 or new_orig_y >= crop_height):
                    # ?꾩쟾??諛뽰뿉 ?덉쑝硫??쒖쇅
                    continue
                
                # Crop ?곸뿭?쇰줈 bitmap ?먮Ⅴ湲?
                x_start = max(0, -new_orig_x)
                y_start = max(0, -new_orig_y)
                x_end = min(bitmap_w, crop_width - new_orig_x)
                y_end = min(bitmap_h, crop_height - new_orig_y)
                
                cropped_bitmap = bitmap_mask[y_start:y_end, x_start:x_end]
                
                # 鍮?bitmap?대㈃ ?쒖쇅
                if cropped_bitmap.size == 0:
                    continue
                
                # Origin 議곗젙 (crop?쇰줈 ?명븳 蹂??
                final_orig_x = max(0, new_orig_x)
                final_orig_y = max(0, new_orig_y)
                
                # ?낅뜲?댄듃??object
                updated_obj = obj.copy()
                updated_obj['bitmap'] = {
                    'data': self.encode_bitmap(cropped_bitmap),
                    'origin': [final_orig_x, final_orig_y]
                }
                
                updated_objects.append(updated_obj)
            
            ann_data['objects'] = updated_objects
            
            # ???
            with open(ann_file, 'w') as f:
                json.dump(ann_data, f, indent=2)
            
            updated_annotations[img_name] = {
                'offset': (offset_x, offset_y),
                'crop_size': (crop_width, crop_height),
                'num_objects': len(updated_objects),
                'confidence': confidence
            }

        print(f"Updated {len(updated_annotations)} annotations")
        if missing_original_images:
            print(f"Warning: Original image not found for {len(missing_original_images)} files")
            preview = ", ".join(sorted(missing_original_images)[:10])
            if preview:
                print(f"  Missing examples: {preview}")
            if len(missing_original_images) > 10:
                print("  ...")

        # ??? confidence ?대?吏 由ы룷??
        if len(self.low_confidence_images) > 0:
            print(f"\nWARNING: Low Template Matching Confidence ({len(self.low_confidence_images)} images):")
            for img_name, conf in sorted(self.low_confidence_images, key=lambda x: x[1]):
                print(f"  - {img_name}: {conf:.3f}")
            print("  ??These images may have GT-Pred alignment issues")

        return updated_annotations
    
    def calculate_gt_size(self, bitmap_mask: np.ndarray, pixel_to_real: float = 1.0) -> float:
        """GT ?ш린 怨꾩궛"""
        pixel_count = np.sum(bitmap_mask > 0)
        return pixel_count * (pixel_to_real ** 2)
    
    def calculate_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """??mask 媛?IoU 怨꾩궛"""
        if mask1.shape != mask2.shape:
            raise ValueError(
                "Mask shape mismatch after observation-domain alignment: "
                f"GT={mask1.shape}, prediction={mask2.shape}"
            )
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def hungarian_matching(self, gt_masks: List[np.ndarray], pred_masks: List[np.ndarray],
                          iou_threshold: float = 0.5) -> Tuple[List[Tuple[int, int, float]], Optional[Dict]]:
        """Hungarian algorithm?쇰줈 GT-Pred 留ㅼ묶 (吏꾨떒 ?뺣낫 ?ы븿)"""
        n_gt = len(gt_masks)
        n_pred = len(pred_masks)

        if n_gt == 0 or n_pred == 0:
            return [], None

        # IoU matrix 怨꾩궛
        iou_matrix = np.zeros((n_gt, n_pred))
        for i, gt_mask in enumerate(gt_masks):
            for j, pred_mask in enumerate(pred_masks):
                iou_matrix[i, j] = self.calculate_iou(gt_mask, pred_mask)

        # Cost matrix (maximize IoU = minimize -IoU)
        cost_matrix = -iou_matrix

        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 留ㅼ묶 寃곌낵 (IoU threshold ?곸슜)
        matches = []
        for i, j in zip(row_ind, col_ind):
            if iou_matrix[i, j] >= iou_threshold:
                matches.append((i, j, iou_matrix[i, j]))

        # 吏꾨떒 ?뺣낫 ?섏쭛
        diagnostics = None
        if len(matches) < min(n_gt, n_pred):
            # Near-miss 寃異?(0.3 <= IoU < threshold)
            near_misses = []
            for i in range(n_gt):
                for j in range(n_pred):
                    iou = iou_matrix[i, j]
                    if 0.3 <= iou < iou_threshold:
                        near_misses.append((i, j, iou))

            diagnostics = {
                'iou_matrix': iou_matrix,
                'near_misses': near_misses,
                'n_gt': n_gt,
                'n_pred': n_pred,
                'n_matched': len(matches)
            }

        return matches, diagnostics
    
    def step3_run_sam2_and_compare(self, image_names: set):
        """Step 3: SAM2 ?덉륫 諛?GT? 鍮꾧탳"""
        print("\n=== Step 3: Running SAM2 and comparing with GT ===")

        # 寃곌낵 ???
        all_results = []
        failed_images = []
        segmentation_stats = []  # Per-image segmentation quality

        # ?쒓컖?붿슜 ?곗씠?????
        self.visualization_data = {}

        for img_name in tqdm(image_names, desc="Processing images"):
            try:
                # ?대?吏 濡쒕뱶
                img_files = list(self.dataset_dir.glob(f"{img_name}.*"))
                if not img_files:
                    failed_images.append((img_name, "Image file not found"))
                    continue

                image = self.load_image(str(img_files[0]))

                # ?꾩쿂由?
                preprocessed = self.preprocess_image(image, image_name=img_name)

                # SAM2 ?덉륫 (罹먯떆 吏??
                pred_masks = self.run_sam2_inference(preprocessed, image_name=img_name)

                # GT 濡쒕뱶
                ann_file = self.matched_ann_dir / f"{img_name}.json"
                if not ann_file.exists():
                    failed_images.append((img_name, "Annotation file not found"))
                    continue

                with open(ann_file, 'r') as f:
                    ann_data = json.load(f)

                gt_masks = []
                gt_sizes = []

                for obj in ann_data.get('objects', []):
                    try:
                        bitmap_mask = self.decode_bitmap(obj['bitmap']['data'])
                        orig_x, orig_y = obj['bitmap']['origin']

                        # Full image mask ?앹꽦
                        full_mask = np.zeros((ann_data['size']['height'],
                                             ann_data['size']['width']), dtype=np.uint8)

                        h, w = bitmap_mask.shape
                        full_mask[orig_y:orig_y+h, orig_x:orig_x+w] = bitmap_mask

                        gt_masks.append(full_mask > 0)
                        gt_sizes.append(self.calculate_gt_size(bitmap_mask))
                    except Exception as e:
                        print(f"  WARNING: Failed to decode object in {img_name}: {e}")
                        continue

                # GT ?꾪꽣留? crop ?곸뿭 諛?諛?寃쎄퀎 ?곗튂 ?뺤씤
                crop_shape = tuple(map(int, preprocessed.shape[:2]))
                annotation_shape = (
                    int(ann_data['size']['height']),
                    int(ann_data['size']['width']),
                )
                if annotation_shape != crop_shape:
                    print(
                        f"  [ALIGN] Cropping GT domain {annotation_shape} -> "
                        f"{crop_shape} (bottom/right only)"
                    )
                unexpected_pred_shapes = sorted({
                    tuple(np.asarray(mask).shape)
                    for mask in pred_masks
                    if tuple(np.asarray(mask).shape) != crop_shape
                })
                if unexpected_pred_shapes:
                    raise ValueError(
                        f"Cached SAM mask shape(s) {unexpected_pred_shapes} do not "
                        f"match preprocessed image shape {crop_shape}"
                    )
                gt_masks_filtered, boundary_flags = (
                    self.align_gt_masks_to_observation_domain(gt_masks, crop_shape)
                )

                # ?꾪꽣留곷맂 GT???좏슚???몃뜳??異붿쟻
                valid_gt_indices = []
                filtered_idx = 0
                for orig_idx in range(len(gt_masks)):
                    if filtered_idx < len(gt_masks_filtered):
                        # 留덉뒪?ш? ?꾪꽣留곷릺吏 ?딄퀬 ?⑥븘?덈뒗 寃쎌슦
                        coords = np.argwhere(gt_masks[orig_idx] > 0)
                        if len(coords) > 0:
                            y_center = int(coords[:, 0].mean())
                            x_center = int(coords[:, 1].mean())
                            if (0 <= x_center < crop_shape[1] and 0 <= y_center < crop_shape[0]):
                                valid_gt_indices.append(orig_idx)
                                filtered_idx += 1

                # ?꾪꽣留??듦퀎 異쒕젰
                if len(gt_masks) > len(gt_masks_filtered):
                    print(f"  Filtered {len(gt_masks) - len(gt_masks_filtered)} invalid GT masks")

                # Define GT area on the same aligned observation domain used
                # for IoU. The legacy variables below are reassigned so all
                # downstream exports and diagnostics use the clipped masks.
                gt_sizes = [self.calculate_gt_size(mask) for mask in gt_masks_filtered]
                valid_gt_indices = list(range(len(gt_masks_filtered)))

                # Pred masks ?ш린 怨꾩궛
                pred_sizes = [np.sum(mask) for mask in pred_masks]

                # Hungarian matching (?꾪꽣留곷맂 GT ?ъ슜)
                matches, diagnostics = self.hungarian_matching(gt_masks_filtered, pred_masks)

                # 吏꾨떒 ?뺣낫 ???
                if diagnostics is not None and len(diagnostics['near_misses']) > 0:
                    print(f"  WARNING: {img_name}: {len(diagnostics['near_misses'])} near-misses detected (0.3 <= IoU < 0.5)")

                # Segmentation quality metrics (?꾪꽣留곷맂 GT 湲곗?)
                n_gt = len(gt_masks_filtered)
                n_pred = len(pred_masks)
                n_matched = len(matches)

                precision = n_matched / n_pred if n_pred > 0 else 0
                recall = n_matched / n_gt if n_gt > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                # ?넅 Boundary蹂?GT/Pred ?듦퀎 怨꾩궛
                n_gt_interior = sum(1 for flag in boundary_flags if not flag)
                n_gt_boundary = sum(1 for flag in boundary_flags if flag)

                # Matched pairs?먯꽌 boundary蹂?遺꾨쪟
                matched_gt_indices = set(gt_idx for gt_idx, _, _ in matches)
                n_matched_interior = sum(1 for gt_idx in matched_gt_indices if gt_idx < len(boundary_flags) and not boundary_flags[gt_idx])
                n_matched_boundary = sum(1 for gt_idx in matched_gt_indices if gt_idx < len(boundary_flags) and boundary_flags[gt_idx])

                segmentation_stats.append({
                    'image': img_name,
                    'n_gt': n_gt,
                    'n_pred': n_pred,
                    'n_matched': n_matched,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    # Boundary蹂??듦퀎
                    'n_gt_interior': n_gt_interior,
                    'n_gt_boundary': n_gt_boundary,
                    'n_matched_interior': n_matched_interior,
                    'n_matched_boundary': n_matched_boundary
                })

                # ?쒓컖?붿슜 ?곗씠?????(?꾪꽣留곷맂 GT ?ъ슜)
                self.visualization_data[img_name] = {
                    'image': image,
                    'gt_masks': gt_masks_filtered,
                    'pred_masks': pred_masks,
                    'matches': matches,
                    'diagnostics': diagnostics,
                    'boundary_flags': boundary_flags,  # boundary 遺꾩꽍??
                    'gt_sizes': [gt_sizes[valid_gt_indices[i]] if i < len(valid_gt_indices) else 0 
                                 for i in range(len(gt_masks_filtered))]  # GT ?ш린 ?뺣낫
                }

                # 利됱떆 overlay ?대?吏 ?앹꽦 諛????(filter_overlapping 媛믪뿉 ?곕씪 ?대뜑 遺꾨━)
                viz_dir = self._get_visualization_dir()
                viz_dir.mkdir(exist_ok=True)
                self._create_overlay_image(img_name, self.visualization_data[img_name], viz_dir)
                print(f"  Saved overlay: {img_name}")

                # 寃곌낵 ???(matches??gt_idx???꾪꽣留곷맂 由ъ뒪?몄쓽 ?몃뜳??
                for gt_idx, pred_idx, iou in matches:
                    # ?먮낯 GT ?몃뜳??李얘린
                    orig_gt_idx = valid_gt_indices[gt_idx] if gt_idx < len(valid_gt_indices) else gt_idx
                    is_boundary = boundary_flags[gt_idx] if gt_idx < len(boundary_flags) else False

                    all_results.append({
                        'image': img_name,
                        'gt_idx': gt_idx,
                        'pred_idx': pred_idx,
                        'iou': iou,
                        'gt_size': gt_sizes[orig_gt_idx],
                        'pred_size': pred_sizes[pred_idx],
                        'size_error': abs(gt_sizes[orig_gt_idx] - pred_sizes[pred_idx]),
                        'size_error_pct': abs(gt_sizes[orig_gt_idx] - pred_sizes[pred_idx]) / gt_sizes[orig_gt_idx] * 100,
                        'gt_boundary_flag': is_boundary
                    })

            except Exception as e:
                failed_images.append((img_name, str(e)))
                print(f"\n[ERROR] Failed to process {img_name}: {e}")
                continue

        # ?ㅽ뙣 ?대?吏 由ы룷??
        if failed_images:
            print(f"\nWARNING: Failed to process {len(failed_images)} images:")
            for img_name, reason in failed_images:
                print(f"  - {img_name}: {reason}")

        # Segmentation stats瑜?instance variable濡????
        self.segmentation_stats = pd.DataFrame(segmentation_stats)

        # Keep a stable schema even when there are no matched pairs
        results_columns = [
            'image',
            'gt_idx',
            'pred_idx',
            'iou',
            'gt_size',
            'pred_size',
            'size_error',
            'size_error_pct',
            'gt_boundary_flag',
        ]
        results_df = pd.DataFrame(all_results, columns=results_columns)

        # ?넅 ?먮룞 以묎컙 寃곌낵 ???(Step 3 ?꾨즺 ?? - overlap logic蹂?遺꾨━
        intermediate_path = self._get_intermediate_results_path()
        intermediate_data = {
            'results_df': results_df,
            'segmentation_stats': self.segmentation_stats,
            'visualization_data': self.visualization_data,  # unmatched GT 遺꾩꽍??
            'filter_overlapping': self.filter_overlapping,  # ?ㅼ젙 ?뺣낫 ???
            'preprocess_mode': self.preprocess_mode_tag,
            'use_noise2sr': self.use_noise2sr,
            'noise2sr_epochs': self.noise2sr_epochs,
            'reuse_preprocessed_only': self.reuse_preprocessed_only,
            'segmentation_model': 'SAM 2.1 Hiera Large',
            'sam_implementation': 'sam2.SAM2AutomaticMaskGenerator',
            'pred_iou_thresh': self.pred_iou_thresh,
            'stability_score_thresh': self.stability_score_thresh,
        }
        with open(intermediate_path, 'wb') as f:
            pickle.dump(intermediate_data, f)
        print(f"\nStep 3 results auto-saved: {intermediate_path}")
        print(f"   ??Saved: results_df + segmentation_stats + visualization_data")
        print(f"   ??filter_overlapping={self.filter_overlapping}")
        print(f"   ??preprocessing_mode={self.preprocess_mode_tag}")
        print(f"   ??You can resume from Step 4 if needed")

        if failed_images:
            raise RuntimeError(
                "Size validation is incomplete. Successful images were cached; "
                "rerun the command to retry the failed images listed above."
            )

        return results_df

    def analyze_by_size_bin(self, results_df: pd.DataFrame):
        """?ш린蹂??깅뒫 ?덉젙??遺꾩꽍 (?듦퀎??bin ?뺤쓽)

        Args:
            results_df: 紐⑤뱺 ?대?吏??matched particle pairs (?꾩껜 ?곗씠?곗뀑)

        Returns:
            bin_stats: ?ш린蹂??듦퀎 DataFrame
            bin_info: Bin ?뺤쓽 ?뺣낫 (mean, std, thresholds)
        """
        # Step 1: ?꾩껜 ?곗씠?곗뀑???듦퀎??Bin ?뺤쓽 (-1SD, Mean, +1SD 湲곗?)
        mean_size = results_df['gt_size'].mean()
        std_size = results_df['gt_size'].std()

        # ?뵩 FIX: Ensure bin edges are monotonically increasing and unique
        lower_bound = max(0, mean_size - std_size)  # Prevent negative values
        upper_bound = mean_size + std_size

        # Check if bounds are valid and unique
        if lower_bound >= upper_bound or abs(lower_bound - upper_bound) < 1e-6:
            # Fallback: Use quartiles instead
            lower_bound = results_df['gt_size'].quantile(0.25)
            upper_bound = results_df['gt_size'].quantile(0.75)

        # Ensure lower_bound is not 0 (duplicate with first edge)
        if abs(lower_bound) < 1e-6:
            # Use percentile-based approach for better separation
            lower_bound = results_df['gt_size'].quantile(0.33)
            upper_bound = results_df['gt_size'].quantile(0.67)

        # Final validation: ensure all edges are unique
        bin_edges = [0, lower_bound, upper_bound, np.inf]

        # Remove duplicates by ensuring minimum separation
        min_separation = 1.0  # Minimum 1 pixel짼 separation
        for i in range(1, len(bin_edges) - 1):
            if bin_edges[i] - bin_edges[i-1] < min_separation:
                bin_edges[i] = bin_edges[i-1] + min_separation

        labels = [
            f'Small (<{int(bin_edges[1])}px짼)',
            f'Medium ({int(bin_edges[1])}-{int(bin_edges[2])}px짼)',
            f'Large (>{int(bin_edges[2])}px짼)'
        ]

        # Bin 遺꾨쪟 (?꾩껜 ?곗씠??湲곗?)
        results_df_copy = results_df.copy()
        results_df_copy['size_bin'] = pd.cut(results_df_copy['gt_size'], bins=bin_edges, labels=labels, duplicates='drop')

        # Step 2: 媛?bin蹂??듦퀎 怨꾩궛
        bin_stats = results_df_copy.groupby('size_bin', observed=True).agg({
            'gt_idx': 'count',              # ?낆옄 媛쒖닔
            'size_error': ['mean', 'std'],  # ?덈? ?ㅼ감
            'size_error_pct': ['mean', 'std'],  # ?곷? ?ㅼ감 (MAPE)
            'iou': ['mean', 'std']          # IoU
        }).round(4)

        # Step 3: 而щ읆紐??뺣━
        bin_stats.columns = ['Count', 'MAE', 'MAE_Std', 'MAPE', 'MAPE_Std', 'IoU', 'IoU_Std']

        # Step 4: Bin ?뺤쓽 ?뺣낫
        bin_info = {
            'mean_size': mean_size,
            'std_size': std_size,
            'bin_edges': [lower_bound, upper_bound]  # Actual used bounds
        }

        return bin_stats, bin_info

    def analyze_boundary_effect(self, results_df: pd.DataFrame):
        """寃쎄퀎 ?낆옄 vs ?대? ?낆옄 ?깅뒫 鍮꾧탳

        Args:
            results_df: 紐⑤뱺 ?대?吏??matched particle pairs (?꾩껜 ?곗씠?곗뀑)

        Returns:
            comparison_df: Boundary vs Interior 鍮꾧탳 DataFrame
        """
        # Step 1: ?곗씠??遺꾨━
        boundary = results_df[results_df['gt_boundary_flag'] == True]
        interior = results_df[results_df['gt_boundary_flag'] == False]

        # Step 2: Recall/Precision 怨꾩궛 (segmentation_stats ?ъ슜)
        if (
            hasattr(self, 'segmentation_stats')
            and isinstance(self.segmentation_stats, pd.DataFrame)
            and {'n_gt_interior', 'n_matched_interior', 'n_gt_boundary', 'n_matched_boundary'}.issubset(self.segmentation_stats.columns)
        ):
            # Interior ?듦퀎
            total_gt_interior = self.segmentation_stats['n_gt_interior'].sum()
            total_matched_interior = self.segmentation_stats['n_matched_interior'].sum()
            recall_interior = total_matched_interior / total_gt_interior if total_gt_interior > 0 else 0

            # Boundary ?듦퀎
            total_gt_boundary = self.segmentation_stats['n_gt_boundary'].sum()
            total_matched_boundary = self.segmentation_stats['n_matched_boundary'].sum()
            recall_boundary = total_matched_boundary / total_gt_boundary if total_gt_boundary > 0 else 0

            # Precision 怨꾩궛 (matched 以묒뿉??媛???낆쓽 鍮꾩쑉)
            # Note: Pred?먮뒗 boundary flag媛 ?놁쑝誘濡? matched pairs??GT flag濡?遺꾨쪟
            precision_interior = total_matched_interior / len(interior) if len(interior) > 0 else 0
            precision_boundary = total_matched_boundary / len(boundary) if len(boundary) > 0 else 0
        else:
            # segmentation_stats媛 ?녿뒗 寃쎌슦 (old format)
            recall_interior = recall_boundary = None
            precision_interior = precision_boundary = None
            total_gt_interior = total_gt_boundary = None
            total_matched_interior = total_matched_boundary = None

        # Step 3: 媛?洹몃９ ?듦퀎
        stats = []
        for name, group in [('Interior', interior), ('Boundary', boundary)]:
            stat_dict = {
                'Type': name,
                'Count (matched)': len(group),
                'Mean IoU': group['iou'].mean(),
                'Std IoU': group['iou'].std(),
                'Mean MAPE (%)': group['size_error_pct'].mean(),
                'Std MAPE (%)': group['size_error_pct'].std()
            }

            # Recall/Precision 異붽? (媛?ν븳 寃쎌슦)
            if name == 'Interior' and recall_interior is not None:
                stat_dict['Total GT'] = total_gt_interior
                stat_dict['Recall'] = recall_interior
                stat_dict['Precision'] = precision_interior
            elif name == 'Boundary' and recall_boundary is not None:
                stat_dict['Total GT'] = total_gt_boundary
                stat_dict['Recall'] = recall_boundary
                stat_dict['Precision'] = precision_boundary

            stats.append(stat_dict)

        comparison_df = pd.DataFrame(stats).round(4)
        return comparison_df

    def step4_calculate_statistics(self, results_df: pd.DataFrame):
        """Step 4: ?듦퀎 怨꾩궛 諛?Excel ???(4 Core Metrics)"""
        print("\n=== Step 4: Calculating statistics and saving to Excel ===")

        if len(results_df) == 0:
            print("No results to analyze!")
            return None, None

        required_result_cols = {
            'image', 'gt_idx', 'iou', 'size_error_pct',
            'gt_size', 'pred_size', 'size_error', 'gt_boundary_flag'
        }
        missing_result_cols = sorted(required_result_cols - set(results_df.columns))
        if missing_result_cols:
            print(f"WARNING: results_df missing required columns: {missing_result_cols}")
            print("Skipping Step 4 statistics due to incompatible results format.")
            return None, None

        # Core Metric 1: Recall (GT Detection Rate)
        if (
            hasattr(self, 'segmentation_stats')
            and isinstance(self.segmentation_stats, pd.DataFrame)
            and len(self.segmentation_stats) > 0
            and {'n_gt', 'n_pred', 'n_matched'}.issubset(self.segmentation_stats.columns)
        ):
            total_gt = self.segmentation_stats['n_gt'].sum()
            total_pred = self.segmentation_stats['n_pred'].sum()
            total_matched = self.segmentation_stats['n_matched'].sum()

            overall_recall = total_matched / total_gt if total_gt > 0 else 0

            # Core Metric 2: Extra Detection Ratio
            # = (Unmatched Pred) / (Total GT)
            # = (Total Pred - Total Matched) / Total GT
            unmatched_pred = total_pred - total_matched
            extra_detection_ratio = unmatched_pred / total_gt if total_gt > 0 else 0
        else:
            overall_recall = 0
            extra_detection_ratio = 0
            total_gt = total_pred = total_matched = 0
            unmatched_pred = 0

        # Core Metric 3: Mean IoU
        mean_iou = results_df['iou'].mean()

        # Core Metric 4: MAPE (%)
        mape = results_df['size_error_pct'].mean()

        # Additional analysis
        size_bin_stats, bin_info = self.analyze_by_size_bin(results_df)
        boundary_stats = self.analyze_boundary_effect(results_df)

        # ?듦퀎 ?붿빟 - 4 Core Metrics + Supporting Data
        stats_summary = {
            'Metric': [
                '=== CORE METRICS ===',
                'Recall (GT Detection Rate)',
                'Extra Detection Ratio',
                'Mean IoU',
                'MAPE (%)',
                '',
                '=== COUNTS ===',
                'Total GT Particles',
                'Total Pred Particles',
                'Total Matched Particles',
                'Unmatched Pred Particles',
                '',
                '=== SIZES ===',
                'Mean GT Size (px짼)',
                'Mean Pred Size (px짼)',
                '',
                '=== SIZE BIN DEFINITION ===',
                'Mean Size (px짼)',
                'Std Size (px짼)',
                'Small Bin Threshold (<)',
                'Large Bin Threshold (>)'
            ],
            'Value': [
                '',  # Separator
                overall_recall,
                extra_detection_ratio,
                mean_iou,
                mape,
                '',  # Separator
                '',  # Separator
                total_gt,
                total_pred,
                total_matched,
                unmatched_pred,
                '',  # Separator
                '',  # Separator
                results_df['gt_size'].mean(),
                results_df['pred_size'].mean(),
                '',  # Separator
                '',  # Separator
                bin_info['mean_size'],
                bin_info['std_size'],
                bin_info['bin_edges'][0],
                bin_info['bin_edges'][1]
            ]
        }
        stats_df = pd.DataFrame(stats_summary)

        # Excel ???
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = self.output_dir / f"size_validation_results_{timestamp}.xlsx"

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Sheet 1: Summary statistics
            stats_df.to_excel(writer, sheet_name='Summary', index=False)

            # Sheet 2: Detailed results
            results_df.to_excel(writer, sheet_name='Detailed Results', index=False)

            # Sheet 3: Per-image statistics (4 core metrics)
            if (
                hasattr(self, 'segmentation_stats')
                and isinstance(self.segmentation_stats, pd.DataFrame)
                and {'image', 'n_gt', 'n_pred', 'n_matched'}.issubset(self.segmentation_stats.columns)
            ):
                # Start with segmentation_stats (has n_gt, n_pred, n_matched per image)
                per_image_stats = self.segmentation_stats[['image', 'n_gt', 'n_pred', 'n_matched']].copy()

                # Calculate recall per image
                per_image_stats['recall'] = per_image_stats['n_matched'] / per_image_stats['n_gt']
                per_image_stats['recall'] = per_image_stats['recall'].fillna(0)

                # Calculate extra_detection_ratio per image (vs Total GT)
                per_image_stats['extra_detection_ratio'] = (
                    (per_image_stats['n_pred'] - per_image_stats['n_matched']) / per_image_stats['n_gt']
                )
                per_image_stats['extra_detection_ratio'] = per_image_stats['extra_detection_ratio'].fillna(0)

                # Get mean_iou and mape from results_df
                per_image_from_results = results_df.groupby('image').agg({
                    'iou': 'mean',
                    'size_error_pct': 'mean'
                }).rename(columns={'iou': 'mean_iou', 'size_error_pct': 'mape'})

                # Merge
                per_image_final = per_image_stats.merge(per_image_from_results, on='image', how='left')
                per_image_final = per_image_final.round(4)
                per_image_final.to_excel(writer, sheet_name='Per Image Stats', index=False)
            else:
                # Fallback: simple groupby
                per_image = results_df.groupby('image').agg({
                    'size_error_pct': 'mean',
                    'iou': 'mean',
                    'gt_idx': 'count'
                }).round(4).rename(columns={'size_error_pct': 'mape', 'iou': 'mean_iou', 'gt_idx': 'count'})
                per_image.to_excel(writer, sheet_name='Per Image Stats')

            # Sheet 4: Segmentation Quality per image
            if hasattr(self, 'segmentation_stats') and isinstance(self.segmentation_stats, pd.DataFrame) and len(self.segmentation_stats) > 0:
                self.segmentation_stats.to_excel(writer, sheet_name='Segmentation Quality', index=False)

            # ?넅 Sheet 5: Size Bin Analysis
            size_bin_stats.to_excel(writer, sheet_name='Size Bin Analysis')

            # ?넅 Sheet 6: Boundary Analysis
            boundary_stats.to_excel(writer, sheet_name='Boundary Analysis', index=False)

            pd.DataFrame(
                {
                    'Item': [
                        'Preprocessing_Method',
                        'Preprocessing_Execution_Policy',
                        'Segmentation_Model',
                        'SAM_Implementation',
                        'SAM_Checkpoint',
                        'SAM_Config',
                        'SAM_Predicted_IoU_Threshold',
                        'SAM_Stability_Score_Threshold',
                        'Expected_Image_Count',
                        'Object_Matching_Criterion',
                    ],
                    'Value': [
                        'BM3D + Noise2SR',
                        (
                            'Reuse existing preprocessed images only'
                            if self.reuse_preprocessed_only
                            else 'Reuse cache, generate missing stages'
                        ),
                        'SAM 2.1 Hiera Large',
                        'sam2.SAM2AutomaticMaskGenerator',
                        'checkpoints/sam2.1_hiera_large.pt',
                        'sam2 package resource: configs/sam2.1/sam2.1_hiera_l.yaml',
                        self.pred_iou_thresh,
                        self.stability_score_thresh,
                        self.expected_images,
                        'Hungarian one-to-one mask matching; IoU >= 0.5',
                    ],
                }
            ).to_excel(writer, sheet_name='Run Metadata', index=False)

        publication_metrics_path = None
        if (
            hasattr(self, 'segmentation_stats')
            and isinstance(self.segmentation_stats, pd.DataFrame)
            and not self.segmentation_stats.empty
        ):
            publication_metrics_path = self.generate_publication_metrics(
                results_df=results_df,
                segmentation_stats=self.segmentation_stats,
                n_bootstrap=10000,
                bootstrap_seed=42,
                source_results_excel=excel_path,
            )
        
        print(f"\n=== Validation Results (4 Core Metrics) ===")
        print(f"\n[Core Metric 1] Recall (GT Detection Rate):")
        print(f"   {overall_recall:.4f} ({total_matched}/{total_gt} GT particles detected)")

        print(f"\n[Core Metric 2] Extra Detection Ratio:")
        print(f"   {extra_detection_ratio:.4f} ({unmatched_pred} extra detections / {total_gt} total GT)")
        print(f"   Interpretation: {unmatched_pred} additional particles detected beyond GT")

        print(f"\n[Core Metric 3] Mean IoU (Segmentation Quality):")
        print(f"   {mean_iou:.4f}")

        print(f"\n[Core Metric 4] MAPE (Size Accuracy):")
        print(f"   {mape:.4f}%")

        print(f"\n[Counts]:")
        print(f"   Total GT: {total_gt}, Total Pred: {total_pred}, Matched: {total_matched}")

        print(f"\nSize Bin Analysis (Mean +/- SD):")
        print(f"  Bin definition: {bin_info['mean_size']:.1f} 짹 {bin_info['std_size']:.1f} px짼")
        for idx, row in size_bin_stats.iterrows():
            print(f"  {idx}: n={row['Count']}, MAPE={row['MAPE']:.2f}%, IoU={row['IoU']:.3f}")

        print(f"\nBoundary Effect Analysis:")
        for _, row in boundary_stats.iterrows():
            print(f"  {row['Type']}: n={row['Count (matched)']}, MAPE={row['Mean MAPE (%)']:.2f}%, IoU={row['Mean IoU']:.3f}")

        print(f"\nResults saved to: {excel_path}")
        if publication_metrics_path is not None:
            print(f"Publication metrics saved to: {publication_metrics_path}")

        # ?넅 Save complete analysis data for re-analysis
        self._save_complete_analysis_data(
            results_df=results_df,
            stats_df=stats_df,
            size_bin_stats=size_bin_stats,
            boundary_stats=boundary_stats,
            core_metrics={
                'recall': overall_recall,
                'extra_detection_ratio': extra_detection_ratio,
                'mean_iou': mean_iou,
                'mape': mape,
                'total_gt': total_gt,
                'total_pred': total_pred,
                'total_matched': total_matched,
                'unmatched_pred': unmatched_pred
            }
        )

        return stats_df, excel_path

    @staticmethod
    def _safe_ratio(numerator, denominator):
        """Divide arrays or scalars while returning NaN for a zero denominator."""
        numerator_array = np.asarray(numerator, dtype=float)
        denominator_array = np.asarray(denominator, dtype=float)
        result = np.full(
            np.broadcast_shapes(numerator_array.shape, denominator_array.shape),
            np.nan,
            dtype=float,
        )
        return np.divide(
            numerator_array,
            denominator_array,
            out=result,
            where=denominator_array != 0,
        )

    def generate_publication_metrics(
        self,
        results_df: pd.DataFrame,
        segmentation_stats: pd.DataFrame,
        output_path: Optional[Path] = None,
        n_bootstrap: int = 10000,
        bootstrap_seed: int = 42,
        source_results_excel: Optional[Path] = None,
        source_sam_config_recorded: bool = True,
        source_sam_config_matches: bool = True,
    ) -> Path:
        """Create GT coverage and matched-particle size metrics with image-cluster CIs."""
        required_results = {
            'image', 'gt_size', 'pred_size', 'iou', 'gt_idx', 'pred_idx'
        }
        required_segmentation = {'image', 'n_gt', 'n_pred', 'n_matched'}
        missing_results = sorted(required_results - set(results_df.columns))
        missing_segmentation = sorted(
            required_segmentation - set(segmentation_stats.columns)
        )
        if missing_results:
            raise ValueError(
                f'Matched-particle results are missing columns: {missing_results}'
            )
        if missing_segmentation:
            raise ValueError(
                f'Segmentation statistics are missing columns: {missing_segmentation}'
            )
        if n_bootstrap <= 0:
            raise ValueError('n_bootstrap must be a positive integer')

        matched = results_df.copy()
        detection = segmentation_stats.copy()
        matched['image'] = matched['image'].astype(str)
        detection['image'] = detection['image'].astype(str)
        if detection['image'].duplicated().any():
            duplicate_images = sorted(
                detection.loc[detection['image'].duplicated(False), 'image'].unique()
            )
            raise ValueError(
                'Segmentation Quality contains duplicate image rows: '
                + ', '.join(duplicate_images)
            )
        unmatched_result_images = sorted(
            set(matched['image']) - set(detection['image'])
        )
        if unmatched_result_images:
            raise ValueError(
                'Matched-particle rows have no Segmentation Quality row: '
                + ', '.join(unmatched_result_images)
            )

        for column in ['n_gt', 'n_pred', 'n_matched']:
            detection[column] = pd.to_numeric(detection[column], errors='raise')
        for column in ['gt_size', 'pred_size', 'iou']:
            matched[column] = pd.to_numeric(matched[column], errors='raise')
        if (matched['gt_size'] <= 0).any():
            raise ValueError('GT particle sizes must be positive for percentage errors')

        # Preserve the source workbook order for the independently seeded
        # detection-precision bootstrap used in the manuscript. Other metrics
        # retain the established sorted-image implementation.
        precision_n_pred = detection['n_pred'].to_numpy(dtype=float)
        precision_n_matched = detection['n_matched'].to_numpy(dtype=float)
        detection = detection.sort_values('image').reset_index(drop=True)
        image_ids = detection['image'].tolist()
        n_images = len(image_ids)
        if n_images == 0:
            raise ValueError('No image-level segmentation statistics are available')

        n_gt = detection['n_gt'].to_numpy(dtype=float)
        n_pred = detection['n_pred'].to_numpy(dtype=float)
        n_matched = detection['n_matched'].to_numpy(dtype=float)
        if (
            (n_gt < 0).any()
            or (n_pred < 0).any()
            or (n_matched < 0).any()
            or (n_matched > n_gt).any()
            or (n_matched > n_pred).any()
        ):
            raise ValueError(
                'Expected non-negative counts with n_matched <= n_gt and n_pred'
            )
        per_image_recall = self._safe_ratio(n_matched, n_gt)
        per_image_precision = self._safe_ratio(n_matched, n_pred)
        total_gt = float(n_gt.sum())
        total_pred = float(n_pred.sum())
        total_matched = float(n_matched.sum())
        coverage_estimates = {
            'Object_Precision_Micro': float(
                self._safe_ratio(total_matched, total_pred)
            ),
            'Object_Recall_Micro': float(self._safe_ratio(total_matched, total_gt)),
            'Macro_Per_Image_Recall': float(np.nanmean(per_image_recall)),
        }

        grouped = {name: frame for name, frame in matched.groupby('image', sort=False)}
        cluster = {
            'count': np.zeros(n_images, dtype=float),
            'sum_abs_error': np.zeros(n_images, dtype=float),
            'sum_squared_error': np.zeros(n_images, dtype=float),
            'sum_ape': np.zeros(n_images, dtype=float),
            'sum_smape': np.zeros(n_images, dtype=float),
            'sum_difference': np.zeros(n_images, dtype=float),
            'sum_gt': np.zeros(n_images, dtype=float),
            'sum_pred': np.zeros(n_images, dtype=float),
            'sum_gt_squared': np.zeros(n_images, dtype=float),
            'sum_pred_squared': np.zeros(n_images, dtype=float),
            'sum_gt_pred': np.zeros(n_images, dtype=float),
            'sum_pred_gt_ratio': np.zeros(n_images, dtype=float),
            'sum_iou': np.zeros(n_images, dtype=float),
        }
        per_image_size_rows = []
        for image_index, image_name in enumerate(image_ids):
            frame = grouped.get(image_name)
            if frame is None or frame.empty:
                per_image_size_rows.append(
                    {
                        'image': image_name,
                        'matched_size_count': 0,
                        'size_mae_px2': np.nan,
                        'size_rmse_px2': np.nan,
                        'size_mape_percent': np.nan,
                        'mean_iou': np.nan,
                    }
                )
                continue
            gt = frame['gt_size'].to_numpy(dtype=float)
            pred = frame['pred_size'].to_numpy(dtype=float)
            iou = frame['iou'].to_numpy(dtype=float)
            difference = pred - gt
            absolute_error = np.abs(difference)
            ape = absolute_error / gt * 100.0
            smape = self._safe_ratio(
                200.0 * absolute_error,
                np.abs(gt) + np.abs(pred),
            )
            values = {
                'count': float(len(frame)),
                'sum_abs_error': float(absolute_error.sum()),
                'sum_squared_error': float(np.square(difference).sum()),
                'sum_ape': float(ape.sum()),
                'sum_smape': float(np.nansum(smape)),
                'sum_difference': float(difference.sum()),
                'sum_gt': float(gt.sum()),
                'sum_pred': float(pred.sum()),
                'sum_gt_squared': float(np.square(gt).sum()),
                'sum_pred_squared': float(np.square(pred).sum()),
                'sum_gt_pred': float((gt * pred).sum()),
                'sum_pred_gt_ratio': float((pred / gt).sum()),
                'sum_iou': float(iou.sum()),
            }
            for key, value in values.items():
                cluster[key][image_index] = value
            per_image_size_rows.append(
                {
                    'image': image_name,
                    'matched_size_count': int(len(frame)),
                    'size_mae_px2': float(absolute_error.mean()),
                    'size_rmse_px2': float(np.sqrt(np.square(difference).mean())),
                    'size_mape_percent': float(ape.mean()),
                    'mean_iou': float(iou.mean()),
                }
            )

        gt_all = matched['gt_size'].to_numpy(dtype=float)
        pred_all = matched['pred_size'].to_numpy(dtype=float)
        iou_all = matched['iou'].to_numpy(dtype=float)
        difference_all = pred_all - gt_all
        absolute_error_all = np.abs(difference_all)
        matched_count = len(matched)
        if int(total_matched) != matched_count:
            raise ValueError(
                'Segmentation Quality n_matched total does not match Detailed Results: '
                f'{int(total_matched)} vs {matched_count}'
            )

        def aggregate_size_metrics(aggregate: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            count = aggregate['count']
            mean_gt = self._safe_ratio(aggregate['sum_gt'], count)
            mean_pred = self._safe_ratio(aggregate['sum_pred'], count)
            mae = self._safe_ratio(aggregate['sum_abs_error'], count)
            mean_gt_squared = self._safe_ratio(aggregate['sum_gt_squared'], count)
            mean_pred_squared = self._safe_ratio(aggregate['sum_pred_squared'], count)
            mean_gt_pred = self._safe_ratio(aggregate['sum_gt_pred'], count)
            variance_gt = np.maximum(0.0, mean_gt_squared - np.square(mean_gt))
            variance_pred = np.maximum(0.0, mean_pred_squared - np.square(mean_pred))
            covariance = mean_gt_pred - mean_gt * mean_pred
            pearson = self._safe_ratio(
                covariance,
                np.sqrt(variance_gt * variance_pred),
            )
            ccc = self._safe_ratio(
                2.0 * covariance,
                variance_gt + variance_pred + np.square(mean_gt - mean_pred),
            )
            sst = aggregate['sum_gt_squared'] - self._safe_ratio(
                np.square(aggregate['sum_gt']), count
            )
            return {
                'Size_MAE_px2': mae,
                'Size_RMSE_px2': np.sqrt(
                    self._safe_ratio(aggregate['sum_squared_error'], count)
                ),
                'Size_MAPE_Percent': self._safe_ratio(aggregate['sum_ape'], count),
                'Size_sMAPE_Percent': self._safe_ratio(
                    aggregate['sum_smape'], count
                ),
                'Size_Bias_px2': self._safe_ratio(
                    aggregate['sum_difference'], count
                ),
                'Size_NMAE_Percent_of_Mean_GT': self._safe_ratio(
                    100.0 * mae, mean_gt
                ),
                'Mean_Predicted_to_GT_Size_Ratio': self._safe_ratio(
                    aggregate['sum_pred_gt_ratio'], count
                ),
                'Mean_Matched_IoU': self._safe_ratio(
                    aggregate['sum_iou'], count
                ),
                'Pearson_r_GT_vs_Pred_Size': pearson,
                'Lin_CCC_GT_vs_Pred_Size': ccc,
                'R2_GT_vs_Pred_Size': 1.0 - self._safe_ratio(
                    aggregate['sum_squared_error'], sst
                ),
            }

        aggregate_total = {
            key: np.asarray(value.sum(), dtype=float) for key, value in cluster.items()
        }
        per_image_mean_iou = self._safe_ratio(cluster['sum_iou'], cluster['count'])
        size_estimates = {
            key: float(value) for key, value in aggregate_size_metrics(aggregate_total).items()
        }
        size_estimates['Macro_Per_Image_Mean_IoU'] = float(
            np.nanmean(per_image_mean_iou)
        )
        size_estimates['Median_Absolute_Size_Error_px2'] = float(
            np.median(absolute_error_all)
        )
        iou_q1, iou_median, iou_q3 = np.quantile(
            iou_all, [0.25, 0.50, 0.75]
        )
        size_estimates.update(
            {
                'Matched_IoU_Q1': float(iou_q1),
                'Matched_IoU_Median': float(iou_median),
                'Matched_IoU_Q3': float(iou_q3),
                'Matched_IoU_IQR': float(iou_q3 - iou_q1),
            }
        )
        if matched_count >= 2 and not np.allclose(gt_all, gt_all[0]):
            regression_slope, regression_intercept = np.polyfit(
                gt_all, pred_all, 1
            )
            fitted_pred = regression_slope * gt_all + regression_intercept
            regression_sst = float(np.square(pred_all - pred_all.mean()).sum())
            regression_sse = float(np.square(pred_all - fitted_pred).sum())
            regression_r2 = (
                1.0 - regression_sse / regression_sst
                if regression_sst > 0.0
                else np.nan
            )
        else:
            regression_slope = np.nan
            regression_intercept = np.nan
            regression_r2 = np.nan
        size_estimates.update(
            {
                'OLS_Regression_Slope': float(regression_slope),
                'OLS_Regression_Intercept_px2': float(regression_intercept),
                'OLS_Regression_R2': float(regression_r2),
            }
        )
        size_estimates['Spearman_rho_GT_vs_Pred_Size'] = float(
            pd.Series(gt_all).corr(pd.Series(pred_all), method='spearman')
        )

        rng = np.random.default_rng(bootstrap_seed)
        precision_rng = np.random.default_rng(bootstrap_seed)
        bootstrap_values = {
            key: []
            for key in [
                *aggregate_size_metrics(aggregate_total),
                'Macro_Per_Image_Mean_IoU',
                *coverage_estimates,
            ]
        }
        batch_size = min(250, n_bootstrap)
        generated = 0
        while generated < n_bootstrap:
            current_batch = min(batch_size, n_bootstrap - generated)
            sampled_indices = rng.integers(
                0, n_images, size=(current_batch, n_images)
            )
            aggregate_boot = {
                key: values[sampled_indices].sum(axis=1)
                for key, values in cluster.items()
            }
            size_boot = aggregate_size_metrics(aggregate_boot)
            sampled_mean_iou = per_image_mean_iou[sampled_indices]
            valid_iou_count = np.isfinite(sampled_mean_iou).sum(axis=1)
            size_boot['Macro_Per_Image_Mean_IoU'] = self._safe_ratio(
                np.nansum(sampled_mean_iou, axis=1),
                valid_iou_count,
            )
            for key, values in size_boot.items():
                bootstrap_values[key].append(np.asarray(values, dtype=float))
            sampled_gt = n_gt[sampled_indices]
            sampled_matched = n_matched[sampled_indices]
            precision_sampled_indices = precision_rng.integers(
                0, n_images, size=(current_batch, n_images)
            )
            bootstrap_values['Object_Precision_Micro'].append(
                self._safe_ratio(
                    precision_n_matched[precision_sampled_indices].sum(axis=1),
                    precision_n_pred[precision_sampled_indices].sum(axis=1),
                )
            )
            bootstrap_values['Object_Recall_Micro'].append(
                self._safe_ratio(
                    sampled_matched.sum(axis=1),
                    sampled_gt.sum(axis=1),
                )
            )
            sampled_recall = per_image_recall[sampled_indices]
            valid_recall_count = np.isfinite(sampled_recall).sum(axis=1)
            bootstrap_values['Macro_Per_Image_Recall'].append(
                self._safe_ratio(
                    np.nansum(sampled_recall, axis=1),
                    valid_recall_count,
                )
            )
            generated += current_batch

        confidence_intervals = {}
        for key, batches in bootstrap_values.items():
            values = np.concatenate(batches)
            values = values[np.isfinite(values)]
            confidence_intervals[key] = (
                tuple(np.quantile(values, [0.025, 0.975]))
                if len(values)
                else (np.nan, np.nan)
            )

        size_units = {
            'Size_MAE_px2': 'px^2',
            'Size_RMSE_px2': 'px^2',
            'Size_MAPE_Percent': 'percent',
            'Size_sMAPE_Percent': 'percent',
            'Size_Bias_px2': 'px^2',
            'Size_NMAE_Percent_of_Mean_GT': 'percent',
            'Mean_Predicted_to_GT_Size_Ratio': 'ratio',
            'Mean_Matched_IoU': 'proportion',
            'Macro_Per_Image_Mean_IoU': 'proportion',
            'Pearson_r_GT_vs_Pred_Size': 'correlation',
            'Lin_CCC_GT_vs_Pred_Size': 'agreement coefficient',
            'R2_GT_vs_Pred_Size': 'coefficient of determination',
            'Median_Absolute_Size_Error_px2': 'px^2',
            'Matched_IoU_Q1': 'proportion',
            'Matched_IoU_Median': 'proportion',
            'Matched_IoU_Q3': 'proportion',
            'Matched_IoU_IQR': 'proportion',
            'OLS_Regression_Slope': 'ratio',
            'OLS_Regression_Intercept_px2': 'px^2',
            'OLS_Regression_R2': 'coefficient of determination',
            'Spearman_rho_GT_vs_Pred_Size': 'rank correlation',
        }
        size_rows = []
        for metric, estimate in size_estimates.items():
            ci_low, ci_high = confidence_intervals.get(metric, (np.nan, np.nan))
            has_cluster_ci = metric in confidence_intervals
            if metric == 'Macro_Per_Image_Mean_IoU':
                metric_n = int(np.isfinite(per_image_mean_iou).sum())
                averaging_method = (
                    'mean of per-image matched-particle mean IoU; image bootstrap CI'
                )
            elif metric == 'Mean_Matched_IoU':
                metric_n = matched_count
                averaging_method = 'pooled matched particles; image-cluster CI'
            else:
                metric_n = matched_count
                averaging_method = (
                    'matched particles; image-cluster CI'
                    if has_cluster_ci
                    else 'matched particles; descriptive estimate only'
                )
            size_rows.append(
                {
                    'Metric': metric,
                    'Estimate': estimate,
                    'CI95_Low': ci_low,
                    'CI95_High': ci_high,
                    'Unit': size_units[metric],
                    'N': metric_n,
                    'Averaging_Method': averaging_method,
                }
            )

        coverage_rows = [
            {
                'Metric': 'Object_Precision_Micro',
                'Estimate': coverage_estimates['Object_Precision_Micro'],
                'CI95_Low': confidence_intervals['Object_Precision_Micro'][0],
                'CI95_High': confidence_intervals['Object_Precision_Micro'][1],
                'Unit': 'proportion',
                'N': int(total_pred),
                'Averaging_Method': 'pooled predicted particles; image-cluster bootstrap CI',
            },
            {
                'Metric': 'Object_Recall_Micro',
                'Estimate': coverage_estimates['Object_Recall_Micro'],
                'CI95_Low': confidence_intervals['Object_Recall_Micro'][0],
                'CI95_High': confidence_intervals['Object_Recall_Micro'][1],
                'Unit': 'proportion',
                'N': int(total_gt),
                'Averaging_Method': 'pooled GT particles; image-cluster bootstrap CI',
            },
            {
                'Metric': 'Macro_Per_Image_Recall',
                'Estimate': coverage_estimates['Macro_Per_Image_Recall'],
                'CI95_Low': confidence_intervals['Macro_Per_Image_Recall'][0],
                'CI95_High': confidence_intervals['Macro_Per_Image_Recall'][1],
                'Unit': 'proportion',
                'N': int(np.isfinite(per_image_recall).sum()),
                'Averaging_Method': 'mean across images; image bootstrap CI',
            },
        ]
        per_image_coverage = pd.DataFrame(
            {
                'Image_ID': image_ids,
                'GT_Particle_Count': n_gt.astype(int),
                'Predicted_Particle_Count': n_pred.astype(int),
                'Matched_Particle_Count': n_matched.astype(int),
                'Precision': per_image_precision,
                'Recall': per_image_recall,
            }
        )
        per_image_size = pd.DataFrame(per_image_size_rows).rename(
            columns={
                'image': 'Image_ID',
                'matched_size_count': 'Matched_Particle_Count',
                'size_mae_px2': 'Size_MAE_px2',
                'size_rmse_px2': 'Size_RMSE_px2',
                'size_mape_percent': 'Size_MAPE_Percent',
                'mean_iou': 'Mean_Matched_IoU',
            }
        )
        matched_export = matched.rename(
            columns={
                'image': 'Image_ID',
                'gt_idx': 'GT_Particle_ID',
                'pred_idx': 'Predicted_Particle_ID',
                'gt_size': 'GT_Size_px2',
                'pred_size': 'Predicted_Size_px2',
                'iou': 'IoU',
            }
        ).copy()
        matched_export['Signed_Size_Error_px2'] = (
            matched_export['Predicted_Size_px2'] - matched_export['GT_Size_px2']
        )
        matched_export['Absolute_Size_Error_px2'] = np.abs(
            matched_export['Signed_Size_Error_px2']
        )
        matched_export['Absolute_Percentage_Error'] = (
            matched_export['Absolute_Size_Error_px2']
            / matched_export['GT_Size_px2']
            * 100.0
        )
        matched_export['Symmetric_Absolute_Percentage_Error'] = self._safe_ratio(
            200.0 * matched_export['Absolute_Size_Error_px2'].to_numpy(dtype=float),
            np.abs(matched_export['GT_Size_px2'].to_numpy(dtype=float))
            + np.abs(matched_export['Predicted_Size_px2'].to_numpy(dtype=float)),
        )

        metadata = pd.DataFrame(
            {
                'Item': [
                    'Source_results_excel',
                    'Number_of_images',
                    'Number_of_GT_particles',
                    'Number_of_predicted_particles',
                    'Number_of_matched_particles',
                    'Object_matching_criterion',
                    'Precision_definition',
                    'Recall_definition',
                    'Report_scope',
                    'Size_error_population',
                    'Size_MAE_definition',
                    'Confidence_interval_method',
                    'Bootstrap_resampling_unit',
                    'Bootstrap_resamples',
                    'Bootstrap_seed',
                    'Confidence_level',
                    'Missing_value_policy',
                    'Generated_at',
                ],
                'Value': [
                    str(source_results_excel.resolve()) if source_results_excel else '',
                    n_images,
                    int(total_gt),
                    int(total_pred),
                    matched_count,
                    'Hungarian one-to-one assignment; matched TP requires IoU >= 0.5',
                    'Matched predictions / all predicted particles; unmatched predictions are FP',
                    'Matched GT particles / all GT particles; no FP term',
                    (
                        'Prediction precision, GT-particle recall, and matched-particle '
                        'overlap and projected-area agreement'
                    ),
                    'Existing IoU-matched GT/prediction particle pairs',
                    'Mean absolute difference between predicted and GT mask area in px^2',
                    'Percentile image-cluster bootstrap',
                    'Image',
                    n_bootstrap,
                    bootstrap_seed,
                    0.95,
                    (
                        'No value imputation; undefined matched-particle values '
                        'reported as NaN'
                    ),
                    datetime.now().astimezone().isoformat(timespec='seconds'),
                ],
            }
        )

        expected_image_count = self.expected_images
        publication_ready = bool(
            expected_image_count is not None
            and n_images == expected_image_count
            and matched_count > 0
            and source_sam_config_recorded
            and source_sam_config_matches
        )
        metadata = pd.concat(
            [
                metadata,
                pd.DataFrame(
                    [
                        {
                            'Item': 'Expected_image_count',
                            'Value': expected_image_count,
                        },
                        {
                            'Item': 'Publication_ready',
                            'Value': publication_ready,
                        },
                        {
                            'Item': 'Preprocessing_method',
                            'Value': 'BM3D + Noise2SR',
                        },
                        {
                            'Item': 'Preprocessing_execution_policy',
                            'Value': (
                                'Reuse existing preprocessed images only'
                                if self.reuse_preprocessed_only
                                else 'Reuse cache, generate missing stages'
                            ),
                        },
                        {
                            'Item': 'Segmentation_model',
                            'Value': 'SAM 2.1 Hiera Large',
                        },
                        {
                            'Item': 'SAM_implementation',
                            'Value': 'sam2.SAM2AutomaticMaskGenerator',
                        },
                        {
                            'Item': 'SAM_checkpoint',
                            'Value': 'checkpoints/sam2.1_hiera_large.pt',
                        },
                        {
                            'Item': 'SAM_config',
                            'Value': 'sam2 package resource: configs/sam2.1/sam2.1_hiera_l.yaml',
                        },
                        {
                            'Item': 'SAM_predicted_IoU_threshold',
                            'Value': self.pred_iou_thresh,
                        },
                        {
                            'Item': 'SAM_stability_score_threshold',
                            'Value': self.stability_score_thresh,
                        },
                        {
                            'Item': 'Source_SAM_config_recorded',
                            'Value': bool(source_sam_config_recorded),
                        },
                        {
                            'Item': 'Source_SAM_config_matches_active_settings',
                            'Value': bool(source_sam_config_matches),
                        },
                    ]
                ),
            ],
            ignore_index=True,
        )

        def manuscript_row(item, value, unit, definition):
            return {
                'Item': item,
                'Value': value,
                'Unit': unit,
                'Definition': definition,
            }

        manuscript_rows = [
            manuscript_row(
                'Segmentation_Model',
                'SAM 2.1 Hiera Large',
                'model',
                'SAM2 automatic mask generation; not SAM1',
            ),
            manuscript_row(
                'SAM_Checkpoint',
                'checkpoints/sam2.1_hiera_large.pt',
                'path',
                'SAM 2.1 Hiera Large checkpoint',
            ),
            manuscript_row(
                'Publication_Ready',
                publication_ready,
                'boolean',
                (
                    'True only when the explicit expected image count is met and '
                    'the source SAM thresholds match the active settings'
                ),
            ),
            manuscript_row(
                'Expected_Image_Count',
                expected_image_count,
                'images',
                'Value supplied with --expected-images',
            ),
            manuscript_row(
                'Analyzed_Image_Count', n_images, 'images',
                'Unique rows in Segmentation Quality',
            ),
            manuscript_row(
                'GT_Particle_Count', int(total_gt), 'particles',
                'All GT particles in analyzed images',
            ),
            manuscript_row(
                'Predicted_Particle_Count', int(total_pred), 'particles',
                'All post-processed SAM2 predictions in analyzed images',
            ),
            manuscript_row(
                'Matched_Particle_Count', matched_count, 'particles',
                'Hungarian one-to-one matches with mask IoU >= 0.5',
            ),
            manuscript_row(
                'Object_Precision_Micro',
                coverage_estimates['Object_Precision_Micro'],
                'proportion',
                'Matched predictions divided by all predicted particles',
            ),
            manuscript_row(
                'Object_Precision_Micro_CI95_Low',
                confidence_intervals['Object_Precision_Micro'][0],
                'proportion',
                'Percentile image-cluster bootstrap',
            ),
            manuscript_row(
                'Object_Precision_Micro_CI95_High',
                confidence_intervals['Object_Precision_Micro'][1],
                'proportion',
                'Percentile image-cluster bootstrap',
            ),
            manuscript_row(
                'Object_Recall_Micro',
                coverage_estimates['Object_Recall_Micro'],
                'proportion',
                'Matched GT particles divided by all GT particles',
            ),
            manuscript_row(
                'Object_Recall_Micro_CI95_Low',
                confidence_intervals['Object_Recall_Micro'][0],
                'proportion',
                'Percentile image-cluster bootstrap',
            ),
            manuscript_row(
                'Object_Recall_Micro_CI95_High',
                confidence_intervals['Object_Recall_Micro'][1],
                'proportion',
                'Percentile image-cluster bootstrap',
            ),
            manuscript_row(
                'Macro_Per_Image_Recall',
                coverage_estimates['Macro_Per_Image_Recall'],
                'proportion',
                'Arithmetic mean of per-image GT-particle recall',
            ),
            manuscript_row(
                'Macro_Per_Image_Recall_CI95_Low',
                confidence_intervals['Macro_Per_Image_Recall'][0],
                'proportion',
                'Percentile image bootstrap',
            ),
            manuscript_row(
                'Macro_Per_Image_Recall_CI95_High',
                confidence_intervals['Macro_Per_Image_Recall'][1],
                'proportion',
                'Percentile image bootstrap',
            ),
            manuscript_row(
                'Mean_Matched_IoU', size_estimates['Mean_Matched_IoU'],
                'proportion', 'Pooled one-to-one matched particles',
            ),
            manuscript_row(
                'Mean_Matched_IoU_CI95_Low',
                confidence_intervals['Mean_Matched_IoU'][0],
                'proportion',
                'Percentile image-cluster bootstrap',
            ),
            manuscript_row(
                'Mean_Matched_IoU_CI95_High',
                confidence_intervals['Mean_Matched_IoU'][1],
                'proportion',
                'Percentile image-cluster bootstrap',
            ),
            manuscript_row(
                'Matched_IoU_Q1', size_estimates['Matched_IoU_Q1'],
                'proportion', '25th percentile across matched particles',
            ),
            manuscript_row(
                'Matched_IoU_Median', size_estimates['Matched_IoU_Median'],
                'proportion', 'Median across matched particles',
            ),
            manuscript_row(
                'Matched_IoU_Q3', size_estimates['Matched_IoU_Q3'],
                'proportion', '75th percentile across matched particles',
            ),
            manuscript_row(
                'Matched_IoU_IQR', size_estimates['Matched_IoU_IQR'],
                'proportion', 'Matched_IoU_Q3 minus Matched_IoU_Q1',
            ),
            manuscript_row(
                'Size_MAE_px2', size_estimates['Size_MAE_px2'], 'px^2',
                'Mean absolute area error across matched particles',
            ),
            manuscript_row(
                'Size_MAE_CI95_Low', confidence_intervals['Size_MAE_px2'][0],
                'px^2', 'Percentile image-cluster bootstrap',
            ),
            manuscript_row(
                'Size_MAE_CI95_High', confidence_intervals['Size_MAE_px2'][1],
                'px^2', 'Percentile image-cluster bootstrap',
            ),
            manuscript_row(
                'Identity_Line_R2', size_estimates['R2_GT_vs_Pred_Size'],
                'coefficient of determination',
                '1 - sum((predicted-GT)^2) / sum((GT-mean_GT)^2)',
            ),
            manuscript_row(
                'OLS_Regression_Slope', size_estimates['OLS_Regression_Slope'],
                'ratio', 'OLS regression of predicted area on GT area',
            ),
            manuscript_row(
                'OLS_Regression_Intercept_px2',
                size_estimates['OLS_Regression_Intercept_px2'],
                'px^2', 'OLS regression of predicted area on GT area',
            ),
            manuscript_row(
                'OLS_Regression_R2', size_estimates['OLS_Regression_R2'],
                'coefficient of determination',
                'R^2 of the fitted OLS regression of predicted area on GT area',
            ),
            manuscript_row(
                'SAM_Predicted_IoU_Threshold', self.pred_iou_thresh,
                'proportion', 'Active SAM2 automatic-mask-generator setting',
            ),
            manuscript_row(
                'SAM_Stability_Score_Threshold', self.stability_score_thresh,
                'proportion', 'Active SAM2 automatic-mask-generator setting',
            ),
        ]
        manuscript_values = pd.DataFrame(
            manuscript_rows, columns=['Item', 'Value', 'Unit', 'Definition']
        )

        output_path = Path(
            output_path or self.output_dir / 'size_validation_publication_metrics.xlsx'
        ).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            pd.DataFrame(coverage_rows).to_excel(
                writer, sheet_name='Coverage_Summary', index=False
            )
            pd.DataFrame(size_rows).to_excel(
                writer, sheet_name='Size_Error_Summary', index=False
            )
            per_image_coverage.to_excel(
                writer, sheet_name='Per_Image_Coverage', index=False
            )
            per_image_size.to_excel(
                writer, sheet_name='Per_Image_Size', index=False
            )
            matched_export.to_excel(
                writer, sheet_name='Matched_Particles', index=False
            )
            manuscript_values.to_excel(
                writer, sheet_name='Manuscript Values', index=False
            )
            metadata.to_excel(writer, sheet_name='Metadata', index=False)

        print("\nPublication-ready size validation statistics")
        print(f"  Images: {n_images}")
        print(f"  Matched particles: {matched_count}")
        print(f"  Object precision (micro): {coverage_estimates['Object_Precision_Micro']:.6f}")
        print(f"  Object recall (micro): {coverage_estimates['Object_Recall_Micro']:.6f}")
        print(
            f"  Macro per-image matched IoU: "
            f"{size_estimates['Macro_Per_Image_Mean_IoU']:.6f}"
        )
        print(f"  Size MAE: {size_estimates['Size_MAE_px2']:.6f} px^2")
        print(f"  Size RMSE: {size_estimates['Size_RMSE_px2']:.6f} px^2")
        print(f"  Bootstrap: {n_bootstrap} image-level resamples (seed={bootstrap_seed})")
        print(f"  Publication ready: {publication_ready}")
        print(f"  Saved: {output_path}")
        return output_path

    def generate_publication_metrics_from_excel(
        self,
        results_excel_path: str,
        output_path: Optional[Path] = None,
        n_bootstrap: int = 10000,
        bootstrap_seed: int = 42,
    ) -> Path:
        """Generate publication metrics from existing Excel without model inference."""
        source_path = self._resolve_results_excel_path(results_excel_path)
        xls = pd.ExcelFile(source_path)
        required_sheets = {'Detailed Results', 'Segmentation Quality'}
        missing_sheets = sorted(required_sheets - set(xls.sheet_names))
        if missing_sheets:
            raise ValueError(
                f'Results Excel is missing required sheets: {missing_sheets}'
            )
        results_df = pd.read_excel(
            source_path, sheet_name='Detailed Results', dtype={'image': str}
        )
        segmentation_stats = pd.read_excel(
            source_path, sheet_name='Segmentation Quality', dtype={'image': str}
        )
        source_sam_config_recorded = False
        source_sam_config_matches = False
        if 'Run Metadata' in xls.sheet_names:
            run_metadata = pd.read_excel(source_path, sheet_name='Run Metadata')
            if {'Item', 'Value'}.issubset(run_metadata.columns):
                run_values = {
                    str(row['Item']).strip(): row['Value']
                    for _, row in run_metadata.iterrows()
                }
                recorded_pred_iou = pd.to_numeric(
                    run_values.get('SAM_Predicted_IoU_Threshold'),
                    errors='coerce',
                )
                recorded_stability = pd.to_numeric(
                    run_values.get('SAM_Stability_Score_Threshold'),
                    errors='coerce',
                )
                source_sam_config_recorded = bool(
                    np.isfinite(recorded_pred_iou)
                    and np.isfinite(recorded_stability)
                )
                source_sam_config_matches = bool(
                    source_sam_config_recorded
                    and np.isclose(
                        float(recorded_pred_iou), self.pred_iou_thresh,
                        rtol=0.0, atol=1e-12,
                    )
                    and np.isclose(
                        float(recorded_stability), self.stability_score_thresh,
                        rtol=0.0, atol=1e-12,
                    )
                )
                if self.expected_images is None:
                    recorded_expected = pd.to_numeric(
                        run_values.get('Expected_Image_Count'), errors='coerce'
                    )
                    if np.isfinite(recorded_expected):
                        self.expected_images = int(recorded_expected)
        if not source_sam_config_recorded:
            print(
                '[WARN] Source size workbook has no SAM threshold provenance; '
                'Publication_Ready will be False'
            )
        elif not source_sam_config_matches:
            print(
                '[WARN] Source size workbook SAM thresholds do not match the '
                'active settings; Publication_Ready will be False'
            )
        self.segmentation_stats = segmentation_stats
        return self.generate_publication_metrics(
            results_df=results_df,
            segmentation_stats=segmentation_stats,
            output_path=output_path,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
            source_results_excel=source_path,
            source_sam_config_recorded=source_sam_config_recorded,
            source_sam_config_matches=source_sam_config_matches,
        )

    def step5_visualize_results(self, results_df: pd.DataFrame):
        """Step 5: Generate statistical visualizations."""
        print("\n=== Step 5: Generating statistical visualizations ===")

        viz_dir = self._get_visualization_dir()
        viz_dir.mkdir(exist_ok=True)

        if not hasattr(self, 'visualization_data') or self.visualization_data is None:
            self.visualization_data = {}

        required_metric_cols = {
            'image', 'gt_idx', 'iou', 'size_error_pct',
            'gt_size', 'pred_size', 'size_error', 'gt_boundary_flag'
        }
        results_cols = set(results_df.columns) if isinstance(results_df, pd.DataFrame) else set()
        missing_metric_cols = sorted(required_metric_cols - results_cols)
        has_metric_data = (
            isinstance(results_df, pd.DataFrame)
            and not results_df.empty
            and len(missing_metric_cols) == 0
        )

        # 1. 4 Core Metrics Dashboard (Main)
        print("  Creating 4-metric dashboard...")
        self._create_4_metric_dashboard(results_df, viz_dir)

        if not has_metric_data:
            if isinstance(results_df, pd.DataFrame) and results_df.empty:
                print("  WARNING: No matched pairs in results_df; skipping metric charts.")
            elif missing_metric_cols:
                print(f"  WARNING: Missing metric columns in results_df: {missing_metric_cols}")
                print("  Skipping metric charts that require matched-pair metrics.")
            else:
                print("  WARNING: Invalid results_df; skipping metric charts.")

            print("  Creating unmatched GT analysis...")
            self._create_unmatched_gt_analysis(viz_dir)

            print("  Creating diagnostic visualizations...")
            self._create_diagnostic_visualizations(viz_dir)

            print(f"\n[SUCCESS] Visualizations saved to: {viz_dir}")
            print("   (Metric charts were skipped due to missing matched-pair data)")
            return viz_dir

        # 2. Publication figure (1x2): GT vs Pred + IoU distribution
        print("  Creating publication 1x2 figure (size agreement + IoU)...")
        self._create_publication_size_iou_figure(results_df, viz_dir)

        # 3. MAPE distribution histogram
        print("  Creating MAPE distribution...")
        self._create_error_distribution(results_df, viz_dir)

        # 4. Size bin analysis
        print("  Creating size bin analysis...")
        self._create_size_bin_chart(results_df, viz_dir)

        # 5. Boundary effect comparison
        print("  Creating boundary effect comparison...")
        self._create_boundary_comparison_chart(results_df, viz_dir)

        # 6. Unmatched GT analysis
        print("  Creating unmatched GT analysis...")
        self._create_unmatched_gt_analysis(viz_dir)

        # 7. Diagnostic visualizations (near-misses)
        print("  Creating diagnostic visualizations...")
        self._create_diagnostic_visualizations(viz_dir)

        print(f"\n[SUCCESS] All visualizations saved to: {viz_dir}")
        print("   (Per-image overlays were saved during processing)")
        return viz_dir

    def _save_info_figure(self, output_path: Path, title: str, message: str):
        """Save a simple text figure for graceful fallback reporting."""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.text(
            0.5, 0.5, message,
            ha='center', va='center',
            fontsize=11, fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='whitesmoke', alpha=0.9)
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _create_no_match_dashboard(self, results_df: Optional[pd.DataFrame],
                                   viz_dir: Path, missing_cols: Optional[List[str]] = None):
        """Create dashboard fallback when matched-pair metrics are unavailable."""
        seg_stats = self._get_valid_segmentation_stats(
            required_cols=['n_gt', 'n_pred', 'n_matched'],
            context="_create_no_match_dashboard"
        )
        total_gt = int(seg_stats['n_gt'].sum()) if seg_stats is not None else 0
        total_pred = int(seg_stats['n_pred'].sum()) if seg_stats is not None else 0
        total_matched = int(seg_stats['n_matched'].sum()) if seg_stats is not None else 0

        n_rows = int(len(results_df)) if isinstance(results_df, pd.DataFrame) else 0
        columns = list(results_df.columns) if isinstance(results_df, pd.DataFrame) else []
        missing_text = ", ".join(missing_cols) if missing_cols else "None"
        col_preview = ", ".join(columns[:12]) if columns else "(none)"
        if len(columns) > 12:
            col_preview += ", ..."

        message = (
            "Matched-pair metric data is unavailable.\n\n"
            f"rows in results_df: {n_rows}\n"
            f"total_gt (segmentation_stats): {total_gt}\n"
            f"total_pred (segmentation_stats): {total_pred}\n"
            f"total_matched (segmentation_stats): {total_matched}\n\n"
            f"missing columns: {missing_text}\n"
            f"results_df columns: {col_preview}\n\n"
            "Run Step 3 again if you expected matched pairs."
        )
        self._save_info_figure(viz_dir / "4_metric_dashboard.png", "4-Metric Dashboard", message)

    def _draw_gt_only(self, image: np.ndarray, gt_masks: List[np.ndarray]) -> np.ndarray:
        """GT ?꾩슜 ?쒓컖??(紐⑤뱺 GT瑜??뚮??됱쑝濡??쒖떆)"""
        # RGB濡?蹂??
        if len(image.shape) == 2:
            rgb_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb_img = image.copy()

        # 紐⑤뱺 GT masks瑜?cyan?쇰줈 ?쒖떆
        for mask in gt_masks:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb_img, contours, -1, (255, 255, 0), 2)  # BGR: Cyan

        return rgb_img

    def _draw_matching_results(self, image: np.ndarray, gt_masks: List[np.ndarray],
                               pred_masks: List[np.ndarray], matches: List[Tuple[int, int, float]]) -> np.ndarray:
        """留ㅼ묶 寃곌낵 ?쒓컖??(Matched GT=Green, Unmatched GT=Red, Unmatched Pred=Blue)"""
        # RGB濡?蹂??
        if len(image.shape) == 2:
            rgb_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb_img = image.copy()

        # Matched indices
        matched_gt = set(m[0] for m in matches)
        matched_pred = set(m[1] for m in matches)

        # GT masks: Matched=Green, Unmatched=Red
        for idx, mask in enumerate(gt_masks):
            if idx in matched_gt:
                color = (0, 255, 0)  # BGR: Green (Matched GT)
            else:
                color = (0, 0, 255)  # BGR: Red (Unmatched GT - FN)

            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb_img, contours, -1, color, 2)

        # Unmatched Pred masks: Blue
        for idx, mask in enumerate(pred_masks):
            if idx not in matched_pred:
                color = (255, 0, 0)  # BGR: Blue (Unmatched Pred - FP)
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rgb_img, contours, -1, color, 2)

        return rgb_img

    def _create_colored_legend(self, width: int) -> np.ndarray:
        """?됱긽?붾맂 踰붾? ?앹꽦"""
        legend_h = 100
        legend = np.ones((legend_h, width, 3), dtype=np.uint8) * 255

        # ?띿뒪?몃? 媛??됱긽??留욎떠 ?쒖떆
        cv2.putText(legend, "Green: Matched GT (TP)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(legend, "Red: Unmatched GT (FN)", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(legend, "Blue: Unmatched Pred (FP)", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        return legend

    def _create_overlay_image(self, img_name: str, data: dict, viz_dir: Path):
        """Create side-by-side overlay (GT-only vs matching results)."""
        image = data['image']
        gt_masks = data['gt_masks']
        pred_masks = data['pred_masks']
        matches = data['matches']

        # ?쇱そ: GT ?꾩슜
        gt_only = self._draw_gt_only(image, gt_masks)

        # ?ㅻⅨ履? 留ㅼ묶 寃곌낵
        matching_result = self._draw_matching_results(image, gt_masks, pred_masks, matches)

        # Side-by-side 寃고빀
        combined = np.hstack([gt_only, matching_result])

        # ?됱긽?붾맂 踰붾? 異붽?
        legend = self._create_colored_legend(combined.shape[1])

        # 理쒖쥌 ?대?吏
        result = np.vstack([combined, legend])

        cv2.imwrite(str(viz_dir / f"{img_name}_overlay.png"), result)

    def _create_publication_size_iou_figure(self, results_df: pd.DataFrame, viz_dir: Path):
        """
        Publication-ready 1x2 figure.
        (a) GT vs Pred area scatter + y=x + regression + key metrics
        (b) IoU histogram + mean line + IoU summary stats
        """
        required_cols = {'gt_size', 'pred_size', 'iou'}
        missing_cols = sorted(required_cols - set(results_df.columns))
        output_path = viz_dir / "figure_size_iou_1x2.png"
        output_notext_path = viz_dir / "figure_size_iou_1x2_notext.png"

        if missing_cols:
            message = "Missing required columns:\n" + ", ".join(missing_cols)
            self._save_info_figure(output_path, "Size/IoU Publication Figure", message)
            return

        gt = pd.to_numeric(results_df['gt_size'], errors='coerce').to_numpy(dtype=float)
        pred = pd.to_numeric(results_df['pred_size'], errors='coerce').to_numpy(dtype=float)
        iou_all = pd.to_numeric(results_df['iou'], errors='coerce').to_numpy(dtype=float)

        pair_mask = np.isfinite(gt) & np.isfinite(pred)
        gt = gt[pair_mask]
        pred = pred[pair_mask]
        iou_all = iou_all[np.isfinite(iou_all)]
        iou = iou_all

        if len(gt) == 0 or len(iou_all) == 0:
            self._save_info_figure(
                output_path,
                "Size/IoU Publication Figure",
                "No valid matched-pair data available.",
            )
            return

        n_points = int(len(gt))

        # Regression metrics
        if n_points >= 2 and np.unique(gt).size > 1:
            slope, intercept = np.polyfit(gt, pred, 1)
            pred_fit = slope * gt + intercept
            ss_res = float(np.sum((pred - pred_fit) ** 2))
            ss_tot = float(np.sum((pred - np.mean(pred)) ** 2))
            r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else np.nan
        else:
            slope, intercept, r2 = np.nan, np.nan, np.nan

        mae = float(np.mean(np.abs(pred - gt)))
        rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
        mean_gt = float(np.mean(gt))
        nmae_pct = (mae / mean_gt * 100.0) if mean_gt > 0 else np.nan
        nrmse_pct = (rmse / mean_gt * 100.0) if mean_gt > 0 else np.nan

        # IoU metrics (full matched IoU distribution)
        mean_iou = float(np.mean(iou))
        median_iou = float(np.median(iou))
        n_bins = max(15, min(20, int(np.sqrt(max(len(iou), 1)) * 0.9)))
        iou_ge_050_pct = float(np.mean(iou >= 0.50) * 100.0)
        iou_ge_075_pct = float(np.mean(iou >= 0.75) * 100.0)

        fig, (ax_a, ax_b) = plt.subplots(
            1, 2, figsize=(15, 6), gridspec_kw={'wspace': 0.28}
        )

        # Panel (a): GT vs Pred area
        # Make points stand out more across sparse/dense cases.
        if n_points <= 1500:
            point_size = 40
            point_alpha = 0.86
        elif n_points <= 5000:
            point_size = 30
            point_alpha = 0.78
        else:
            point_size = 22
            point_alpha = 0.68
        ax_a.scatter(
            gt,
            pred,
            s=point_size,
            alpha=point_alpha,
            color='#1F77B4',
            edgecolors='black',
            linewidths=0.45,
            zorder=3,
        )

        axis_min = float(min(np.min(gt), np.min(pred)))
        axis_max = float(max(np.max(gt), np.max(pred)))
        if axis_max <= axis_min:
            axis_max = axis_min + 1.0
        pad = (axis_max - axis_min) * 0.03
        low = max(0.0, axis_min - pad)
        high = axis_max + pad

        # y = x reference
        ax_a.plot([low, high], [low, high], linestyle='--', color='0.55', linewidth=1.2,
                  label='Perfect agreement', zorder=1)

        # Regression line
        if np.isfinite(slope) and np.isfinite(intercept):
            x_line = np.array([low, high], dtype=float)
            y_line = slope * x_line + intercept
            ax_a.plot(x_line, y_line, color='black', linewidth=2.0, label='Regression', zorder=2)

        ax_a.set_xlim(low, high)
        ax_a.set_ylim(low, high)
        ax_a.set_aspect('equal', adjustable='box')
        ax_a.set_xlabel('GT area (pixel$^2$)', fontsize=12)
        ax_a.set_ylabel('Predicted area (pixel$^2$)', fontsize=12)
        ax_a.set_title('(a) GT vs Pred area', fontsize=13, fontweight='bold')
        ax_a.set_axisbelow(True)
        ax_a.grid(True, alpha=0.25)
        ax_a.legend(loc='lower right', fontsize=10, frameon=True)

        def _fmt(value: float, fmt: str) -> str:
            return format(value, fmt) if np.isfinite(value) else "N/A"

        metrics_text = (
            f"N = {n_points}\n"
            f"R$^2$ = {_fmt(r2, '.4f')}\n"
            f"NMAE = {_fmt(nmae_pct, '.2f')}%\n"
            f"NRMSE = {_fmt(nrmse_pct, '.2f')}%"
        )
        ax_a.text(
            0.98, 0.98, metrics_text,
            transform=ax_a.transAxes,
            ha='right', va='top',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='0.55', alpha=0.95),
        )

        # Panel (b): IoU histogram (all matched pairs)
        ax_b.hist(iou, bins=n_bins, edgecolor='white', linewidth=0.8, alpha=0.9, color='#58A65D')
        ax_b.axvline(mean_iou, color='#BF5065', linestyle='--', linewidth=2.0,
                     label=f'Mean = {mean_iou:.3f}')
        # Matching threshold reference (thin)
        ax_b.axvline(0.50, color='0.5', linestyle='--', linewidth=0.9, alpha=0.6, label='IoU = 0.50')
        ax_b.set_xlim(max(0.0, min(0.45, float(np.min(iou)) - 0.02)), 1.0)
        ax_b.set_xlabel('IoU', fontsize=12)
        ax_b.set_ylabel('Frequency', fontsize=12)
        ax_b.set_title('(b) IoU distribution', fontsize=13, fontweight='bold')
        ax_b.grid(True, axis='y', alpha=0.25)
        ax_b.legend(loc='upper left', fontsize=10, frameon=True)

        iou_text = (
            f"N = {len(iou)}\n"
            f"Mean IoU = {_fmt(mean_iou, '.3f')}\n"
            f"Median IoU = {_fmt(median_iou, '.3f')}\n"
            f"IoU >= 0.50 : {iou_ge_050_pct:.1f}%\n"
            f"IoU >= 0.75 : {iou_ge_075_pct:.1f}%"
        )
        ax_b.text(
            0.98, 0.98, iou_text,
            transform=ax_b.transAxes,
            ha='right', va='top',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='0.55', alpha=0.95),
        )

        plt.tight_layout(rect=[0, 0.04, 1, 1])
        fig.savefig(output_path, dpi=200, bbox_inches='tight')

        # Save no-text version: keep axes/ticks/grid/lines, remove text only.
        for fig_text in fig.texts[:]:
            fig_text.set_visible(False)
        for ax in fig.get_axes():
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
            ax.set_title('')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(
                axis='x',
                labelbottom=False,
                labeltop=False,
                bottom=True,
                top=False,
            )
            ax.tick_params(
                axis='y',
                labelleft=False,
                labelright=False,
                left=True,
                right=False,
            )
            for txt in ax.texts[:]:
                txt.remove()
        fig.savefig(output_notext_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

    def _create_scatter_plot(self, results_df: pd.DataFrame, viz_dir: Path):
        """GT size vs Pred size scatter plot"""
        plt.figure(figsize=(10, 8))

        # Scatter plot
        plt.scatter(results_df['gt_size'], results_df['pred_size'], alpha=0.6, s=50, color='#2196F3')

        # Perfect prediction line (identity line)
        max_val = max(results_df['gt_size'].max(), results_df['pred_size'].max())
        min_val = min(results_df['gt_size'].min(), results_df['pred_size'].min())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction (y=x)', linewidth=2)

        plt.xlabel('Ground Truth Size (pixel짼)', fontsize=13)
        plt.ylabel('Predicted Size (pixel짼)', fontsize=13)
        plt.title('GT vs Pred Size Comparison', fontsize=15, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(viz_dir / "scatter_gt_vs_pred.png", dpi=150)
        plt.close()

    def _create_error_distribution(self, results_df: pd.DataFrame, viz_dir: Path):
        """MAPE distribution histogram (Core Metric 4)"""
        plt.figure(figsize=(10, 6))

        # MAPE (percentage error)
        plt.hist(results_df['size_error_pct'], bins=30, edgecolor='black', alpha=0.7, color='#9C27B0')
        plt.axvline(results_df['size_error_pct'].mean(), color='r', linestyle='--', linewidth=2,
                    label=f'Mean MAPE = {results_df["size_error_pct"].mean():.2f}%')

        plt.xlabel('MAPE - Mean Absolute Percentage Error (%)', fontsize=13)
        plt.ylabel('Frequency', fontsize=13)
        plt.title('Size Accuracy Distribution (MAPE)', fontsize=15, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(viz_dir / "mape_distribution.png", dpi=150)
        plt.close()

    def _create_iou_distribution(self, results_df: pd.DataFrame, viz_dir: Path):
        """IoU distribution histogram"""
        plt.figure(figsize=(10, 6))

        plt.hist(results_df['iou'], bins=30, edgecolor='black', alpha=0.7, color='green')
        plt.axvline(results_df['iou'].mean(), color='r', linestyle='--',
                    label=f'Mean IoU = {results_df["iou"].mean():.3f}')
        plt.axvline(0.5, color='blue', linestyle='--', alpha=0.5, label='Threshold = 0.5')

        plt.xlabel('IoU (Intersection over Union)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('IoU Distribution of Matched Particles', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(viz_dir / "iou_distribution.png", dpi=150)
        plt.close()

    def _create_size_bin_chart(self, results_df: pd.DataFrame, viz_dir: Path):
        """?ш린蹂?MAPE bar chart (?듦퀎??bin)"""
        bin_stats, bin_info = self.analyze_by_size_bin(results_df)

        fig, ax = plt.subplots(figsize=(10, 6))

        x = range(len(bin_stats))
        bars = ax.bar(x, bin_stats['MAPE'], yerr=bin_stats['MAPE_Std'],
                      capsize=5, alpha=0.7, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])

        ax.set_xticks(x)
        ax.set_xticklabels(bin_stats.index, rotation=15, ha='right')
        ax.set_ylabel('MAPE (%)', fontsize=12)
        ax.set_title(f'Size Measurement Accuracy by Particle Size\n(Bins: Mean 짹 SD = {bin_info["mean_size"]:.0f} 짹 {bin_info["std_size"]:.0f} px짼)',
                     fontsize=13)
        ax.grid(True, alpha=0.3, axis='y')

        # Count ?쒖떆
        for i, (count, mape, std) in enumerate(zip(bin_stats['Count'], bin_stats['MAPE'], bin_stats['MAPE_Std'])):
            ax.text(i, mape + std + 0.5, f'n={count}', ha='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(viz_dir / "size_bin_accuracy.png", dpi=150)
        plt.close()

    def _create_boundary_comparison_chart(self, results_df: pd.DataFrame, viz_dir: Path):
        """寃쎄퀎 vs ?대? ?낆옄 ?깅뒫 鍮꾧탳"""
        boundary_stats = self.analyze_boundary_effect(results_df)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # IoU 鍮꾧탳
        axes[0].bar(range(2), boundary_stats['Mean IoU'],
                    yerr=boundary_stats['Std IoU'], capsize=5,
                    color=['#FF6B6B', '#4ECDC4'], alpha=0.7)
        axes[0].set_xticks(range(2))
        axes[0].set_xticklabels(boundary_stats['Type'])
        axes[0].set_ylabel('IoU', fontsize=12)
        axes[0].set_title('IoU: Boundary vs Interior', fontsize=13)
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].set_ylim(0, 1)

        # MAPE 鍮꾧탳
        axes[1].bar(range(2), boundary_stats['Mean MAPE (%)'],
                    yerr=boundary_stats['Std MAPE (%)'], capsize=5,
                    color=['#FF6B6B', '#4ECDC4'], alpha=0.7)
        axes[1].set_xticks(range(2))
        axes[1].set_xticklabels(boundary_stats['Type'])
        axes[1].set_ylabel('MAPE (%)', fontsize=12)
        axes[1].set_title('Size Error: Boundary vs Interior', fontsize=13)
        axes[1].grid(True, alpha=0.3, axis='y')

        # Count? 媛??쒖떆
        std_columns = ['Std IoU', 'Std MAPE (%)']
        for ax, metric, std_col in zip(axes, ['Mean IoU', 'Mean MAPE (%)'], std_columns):
            for i, (count, val, std) in enumerate(zip(boundary_stats['Count (matched)'],
                                                       boundary_stats[metric],
                                                       boundary_stats[std_col])):
                # Count
                ax.text(i, val + std + 0.02, f'n={count}', ha='center', fontsize=10, fontweight='bold')
                # 媛?
                ax.text(i, val/2, f'{val:.3f}', ha='center', fontsize=11, color='white', fontweight='bold')

        plt.tight_layout()
        plt.savefig(viz_dir / "boundary_effect_comparison.png", dpi=150)
        plt.close()

    def _create_4_metric_dashboard(self, results_df: pd.DataFrame, viz_dir: Path):
        """4 Core Metrics Dashboard - Graphical Representation"""
        required_cols = {'image', 'iou', 'size_error_pct'}
        if not isinstance(results_df, pd.DataFrame):
            self._create_no_match_dashboard(None, viz_dir, sorted(required_cols))
            return

        missing_cols = sorted(required_cols - set(results_df.columns))
        if results_df.empty or missing_cols:
            self._create_no_match_dashboard(results_df, viz_dir, missing_cols)
            return

        # Calculate overall metrics
        if (
            hasattr(self, 'segmentation_stats')
            and isinstance(self.segmentation_stats, pd.DataFrame)
            and {'n_gt', 'n_pred', 'n_matched'}.issubset(self.segmentation_stats.columns)
        ):
            total_gt = self.segmentation_stats['n_gt'].sum()
            total_pred = self.segmentation_stats['n_pred'].sum()
            total_matched = self.segmentation_stats['n_matched'].sum()

            recall = total_matched / total_gt if total_gt > 0 else 0
            extra_detection_ratio = (total_pred - total_matched) / total_gt if total_gt > 0 else 0
        else:
            recall = 0
            extra_detection_ratio = 0

        mean_iou = results_df['iou'].mean()
        mape = results_df['size_error_pct'].mean()

        # Create figure with 2x2 grid for graphs
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

        # Get per-image data
        if (
            hasattr(self, 'segmentation_stats')
            and isinstance(self.segmentation_stats, pd.DataFrame)
            and {'image', 'n_gt', 'n_pred', 'n_matched'}.issubset(self.segmentation_stats.columns)
        ):
            per_image_stats = self.segmentation_stats[['image', 'n_gt', 'n_pred', 'n_matched']].copy()
            per_image_stats['recall'] = per_image_stats['n_matched'] / per_image_stats['n_gt']
            per_image_stats['extra_detection_ratio'] = (
                (per_image_stats['n_pred'] - per_image_stats['n_matched']) / per_image_stats['n_gt']
            )
            # fillna(0)??NaN留?泥섎━?섎?濡? inf??0?쇰줈 ?泥?
            per_image_stats = per_image_stats.fillna(0).replace([np.inf, -np.inf], 0)

            # Get mean_iou and mape from results_df
            per_image_from_results = results_df.groupby('image').agg({
                'iou': 'mean',
                'size_error_pct': 'mean'
            }).rename(columns={'iou': 'mean_iou', 'size_error_pct': 'mape'})

            # Merge
            per_image_final = per_image_stats.merge(per_image_from_results, on='image', how='left')

            # Sort by recall (descending)
            per_image_final = per_image_final.sort_values('recall', ascending=False)

            # Graph 1: Recall Distribution (top-left)
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.hist(per_image_final['recall'], bins=20, color='#2196F3', alpha=0.7, edgecolor='black')
            ax1.axvline(recall, color='red', linestyle='--', linewidth=2, label=f'Mean: {recall:.3f}')
            ax1.set_xlabel('Recall', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
            ax1.set_title('Recall Distribution (GT Detection Rate)', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)

            # Graph 2: Extra Detection Ratio Distribution (top-right)
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.hist(per_image_final['extra_detection_ratio'], bins=20, color='#FF9800', alpha=0.7, edgecolor='black')
            ax2.axvline(extra_detection_ratio, color='red', linestyle='--', linewidth=2, label=f'Mean: {extra_detection_ratio:.3f}')
            ax2.set_xlabel('Extra Detection Ratio', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
            ax2.set_title('Extra Detection Ratio Distribution (vs Total GT)', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=11)
            ax2.grid(True, alpha=0.3)

            # Graph 3: Mean IoU Distribution (bottom-left)
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.hist(per_image_final['mean_iou'], bins=20, color='#4CAF50', alpha=0.7, edgecolor='black')
            ax3.axvline(mean_iou, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_iou:.3f}')
            ax3.set_xlabel('Mean IoU', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
            ax3.set_title('Mean IoU Distribution (Segmentation Quality)', fontsize=14, fontweight='bold')
            ax3.legend(fontsize=11)
            ax3.grid(True, alpha=0.3)

            # Graph 4: MAPE Distribution (bottom-right)
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.hist(per_image_final['mape'], bins=20, color='#9C27B0', alpha=0.7, edgecolor='black')
            ax4.axvline(mape, color='red', linestyle='--', linewidth=2, label=f'Mean: {mape:.1f}%')
            ax4.set_xlabel('MAPE (%)', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
            ax4.set_title('MAPE Distribution (Size Accuracy)', fontsize=14, fontweight='bold')
            ax4.legend(fontsize=11)
            ax4.grid(True, alpha=0.3)
        else:
            # segmentation_stats媛 ?녿뒗 寃쎌슦: results_df留뚯쑝濡??쒓컖??
            print("    WARNING: No segmentation_stats available, using results_df only")
            
            # Per-image ?듦퀎 怨꾩궛 (results_df?먯꽌)
            per_image_from_results = results_df.groupby('image').agg({
                'iou': 'mean',
                'size_error_pct': 'mean',
                'gt_idx': 'count'
            }).rename(columns={'iou': 'mean_iou', 'size_error_pct': 'mape', 'gt_idx': 'n_matched'})
            
            # Graph 1: IoU Distribution (?泥?
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.hist(per_image_from_results['mean_iou'], bins=20, color='#2196F3', alpha=0.7, edgecolor='black')
            ax1.axvline(mean_iou, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_iou:.3f}')
            ax1.set_xlabel('Mean IoU', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
            ax1.set_title('Mean IoU Distribution (per image)', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)

            # Graph 2: MAPE Distribution
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.hist(per_image_from_results['mape'], bins=20, color='#FF9800', alpha=0.7, edgecolor='black')
            ax2.axvline(mape, color='red', linestyle='--', linewidth=2, label=f'Mean: {mape:.1f}%')
            ax2.set_xlabel('MAPE (%)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
            ax2.set_title('MAPE Distribution (per image)', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=11)
            ax2.grid(True, alpha=0.3)

            # Graph 3: Matched Count Distribution
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.hist(per_image_from_results['n_matched'], bins=20, color='#4CAF50', alpha=0.7, edgecolor='black')
            ax3.axvline(per_image_from_results['n_matched'].mean(), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {per_image_from_results["n_matched"].mean():.1f}')
            ax3.set_xlabel('Matched Particles per Image', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
            ax3.set_title('Matched Particle Count Distribution', fontsize=14, fontweight='bold')
            ax3.legend(fontsize=11)
            ax3.grid(True, alpha=0.3)

            # Graph 4: Summary Text
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.axis('off')
            summary_text = f"""Summary (No segmentation_stats)
            
Total Matched Particles: {len(results_df)}
Total Images: {len(per_image_from_results)}

Mean IoU: {mean_iou:.4f}
MAPE: {mape:.2f}%

Note: Recall & Extra Detection not available
(Run full pipeline to get complete stats)"""
            ax4.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                    fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.savefig(viz_dir / "4_metric_dashboard.png", dpi=150, bbox_inches='tight')
        plt.close()

    def _create_unmatched_gt_analysis(self, viz_dir: Path):
        """?몄떇?섏? 紐삵븳 GT ?낆옄 遺꾩꽍 諛??쒓컖??(Boundary 遺꾩꽍 ?ы븿)"""
        diagnostic_dir = viz_dir / "diagnostics"
        diagnostic_dir.mkdir(exist_ok=True)

        unmatched_dir = diagnostic_dir / "unmatched_gt"
        unmatched_dir.mkdir(exist_ok=True)

        all_unmatched_gt = []  # 紐⑤뱺 unmatched GT ?뺣낫 ?섏쭛
        images_with_unmatched = []  # Unmatched GT媛 ?덈뒗 ?대?吏 由ъ뒪??
        
        # Boundary 遺꾩꽍???듦퀎
        total_gt_boundary = 0
        total_gt_interior = 0
        unmatched_boundary = 0
        unmatched_interior = 0

        for img_name, data in self.visualization_data.items():
            image = data['image']
            gt_masks = data['gt_masks']
            pred_masks = data['pred_masks']
            matches = data['matches']
            boundary_flags = data.get('boundary_flags', [])  # boundary ?뺣낫
            gt_sizes_saved = data.get('gt_sizes', [])  # ??λ맂 GT ?ш린

            # Matched GT indices
            matched_gt_indices = set(m[0] for m in matches)

            # Unmatched GT indices
            unmatched_gt_indices = [i for i in range(len(gt_masks)) if i not in matched_gt_indices]
            
            # Boundary ?듦퀎 ?낅뜲?댄듃
            for i in range(len(gt_masks)):
                is_boundary = boundary_flags[i] if i < len(boundary_flags) else False
                if is_boundary:
                    total_gt_boundary += 1
                    if i in unmatched_gt_indices:
                        unmatched_boundary += 1
                else:
                    total_gt_interior += 1
                    if i in unmatched_gt_indices:
                        unmatched_interior += 1

            if len(unmatched_gt_indices) == 0:
                continue  # 紐⑤뱺 GT媛 留ㅼ묶??寃쎌슦 ?ㅽ궢

            images_with_unmatched.append(img_name)

            # RGB 蹂??
            if len(image.shape) == 2:
                rgb_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                rgb_img = image.copy()

            # Unmatched GT瑜?鍮④컙?됱쑝濡??쒖떆
            for gt_idx in unmatched_gt_indices:
                mask = gt_masks[gt_idx]
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rgb_img, contours, -1, (0, 0, 255), 2)  # BGR: Red

                # GT ?ш린 怨꾩궛
                gt_size = gt_sizes_saved[gt_idx] if gt_idx < len(gt_sizes_saved) else np.sum(mask)
                is_boundary = boundary_flags[gt_idx] if gt_idx < len(boundary_flags) else False

                # Centroid 怨꾩궛 諛??띿뒪???쒖떆
                coords = np.argwhere(mask > 0)
                if len(coords) > 0:
                    y_center = int(coords[:, 0].mean())
                    x_center = int(coords[:, 1].mean())
                    cv2.putText(rgb_img, f"#{gt_idx}: {gt_size}px", (x_center, y_center),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                    all_unmatched_gt.append({
                        'image': img_name,
                        'gt_idx': gt_idx,
                        'gt_size': gt_size,
                        'is_boundary': is_boundary
                    })

            # Matched GT???뱀깋?쇰줈 ?쒖떆
            for gt_idx in matched_gt_indices:
                mask = gt_masks[gt_idx]
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rgb_img, contours, -1, (0, 255, 0), 1)  # BGR: Green (thin)

            # 踰붾? 異붽?
            legend_h = 120
            legend = np.ones((legend_h, rgb_img.shape[1], 3), dtype=np.uint8) * 255
            cv2.putText(legend, f"Unmatched GT: {len(unmatched_gt_indices)} particles (RED)",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(legend, f"Matched GT: {len(matched_gt_indices)} particles (GREEN)",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(legend, f"Total GT: {len(gt_masks)}, Total Pred: {len(pred_masks)}",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            result = np.vstack([rgb_img, legend])
            cv2.imwrite(str(unmatched_dir / f"{img_name}_unmatched.png"), result)

        # ?듦퀎 遺꾩꽍
        if len(all_unmatched_gt) > 0:
            unmatched_df = pd.DataFrame(all_unmatched_gt)

            # Size ?듦퀎
            size_stats = unmatched_df['gt_size'].describe()
            
            # Boundary 遺꾩꽍 怨꾩궛
            miss_rate_boundary = unmatched_boundary / total_gt_boundary if total_gt_boundary > 0 else 0
            miss_rate_interior = unmatched_interior / total_gt_interior if total_gt_interior > 0 else 0

            # Size distribution ?덉뒪?좉렇??+ Boundary 遺꾩꽍 (2x2 ?덉씠?꾩썐)
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # 1. Size distribution (top-left)
            axes[0, 0].hist(unmatched_df['gt_size'], bins=30, edgecolor='black', alpha=0.7, color='red')
            axes[0, 0].axvline(size_stats['mean'], color='blue', linestyle='--', linewidth=2,
                           label=f'Mean = {size_stats["mean"]:.1f}px짼')
            axes[0, 0].axvline(size_stats['50%'], color='green', linestyle='--', linewidth=2,
                           label=f'Median = {size_stats["50%"]:.1f}px짼')
            axes[0, 0].set_xlabel('GT Size (pixel짼)', fontsize=12)
            axes[0, 0].set_ylabel('Frequency', fontsize=12)
            axes[0, 0].set_title(f'Unmatched GT Size Distribution (n={len(unmatched_df)})', fontsize=14)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # 2. Per-image unmatched count (top-right)
            per_image_count = unmatched_df.groupby('image').size().sort_values(ascending=False).head(15)
            axes[0, 1].barh(range(len(per_image_count)), per_image_count.values, color='red', alpha=0.7)
            axes[0, 1].set_yticks(range(len(per_image_count)))
            axes[0, 1].set_yticklabels(per_image_count.index, fontsize=8)
            axes[0, 1].set_xlabel('Unmatched GT Count', fontsize=12)
            axes[0, 1].set_title('Top 15 Images with Most Unmatched GT', fontsize=14)
            axes[0, 1].grid(True, alpha=0.3, axis='x')

            # 3. Boundary vs Interior Miss Rate (bottom-left)
            categories = ['Boundary\n(Edge-touching)', 'Interior\n(Fully inside)']
            miss_rates = [miss_rate_boundary * 100, miss_rate_interior * 100]
            colors = ['#FF5722', '#4CAF50']
            bars = axes[1, 0].bar(categories, miss_rates, color=colors, alpha=0.8, edgecolor='black')
            axes[1, 0].set_ylabel('Miss Rate (%)', fontsize=12, fontweight='bold')
            axes[1, 0].set_title('Unmatched Rate: Boundary vs Interior GT', fontsize=14, fontweight='bold')
            axes[1, 0].set_ylim(0, max(miss_rates) * 1.3 if max(miss_rates) > 0 else 10)
            
            # 媛??쒖떆
            for bar, rate in zip(bars, miss_rates):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                               f'{rate:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3, axis='y')

            # 4. Boundary ?곸꽭 ?듦퀎 (bottom-right)
            axes[1, 1].axis('off')
            
            # ?듦퀎 ?띿뒪??
            stats_text = f"""
?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧??
     BOUNDARY vs INTERIOR ANALYSIS
?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧??

?뱤 BOUNDARY GT (Edge-touching particles)
   Total:     {total_gt_boundary:>6d}
   Unmatched: {unmatched_boundary:>6d}
   Miss Rate: {miss_rate_boundary*100:>6.2f}%

?뱤 INTERIOR GT (Fully inside particles)
   Total:     {total_gt_interior:>6d}
   Unmatched: {unmatched_interior:>6d}
   Miss Rate: {miss_rate_interior*100:>6.2f}%

?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧??
?렞 CONCLUSION:
   {'?좑툘 Boundary particles are HARDER to detect!' if miss_rate_boundary > miss_rate_interior else '??Interior particles are harder to detect.' if miss_rate_interior > miss_rate_boundary else '?∽툘 Both have similar miss rates.'}
   
   Ratio: {miss_rate_boundary/miss_rate_interior:.2f}x {'(Boundary worse)' if miss_rate_boundary > miss_rate_interior else '(Interior worse)' if miss_rate_interior > miss_rate_boundary else ''} 
?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧??
""" if miss_rate_interior > 0 else f"""
?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧??
     BOUNDARY vs INTERIOR ANALYSIS
?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧??

?뱤 BOUNDARY GT: {total_gt_boundary} total, {unmatched_boundary} unmatched
   Miss Rate: {miss_rate_boundary*100:.2f}%

?뱤 INTERIOR GT: {total_gt_interior} total, {unmatched_interior} unmatched
   Miss Rate: {miss_rate_interior*100:.2f}%
?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧?먥븧??
"""
            axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                          fontsize=10, verticalalignment='top', fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

            plt.tight_layout()
            plt.savefig(diagnostic_dir / "unmatched_gt_statistics.png", dpi=150)
            plt.close()

            # Summary 異쒕젰
            print(f"    Found {len(all_unmatched_gt)} unmatched GT particles across {len(images_with_unmatched)} images")
            print(f"    Size statistics:")
            print(f"      Mean: {size_stats['mean']:.1f}px짼, Median: {size_stats['50%']:.1f}px짼")
            print(f"      Min: {size_stats['min']:.1f}px짼, Max: {size_stats['max']:.1f}px짼")
            print(f"    Boundary Analysis:")
            print(f"      Boundary GT: {unmatched_boundary}/{total_gt_boundary} unmatched ({miss_rate_boundary*100:.1f}%)")
            print(f"      Interior GT: {unmatched_interior}/{total_gt_interior} unmatched ({miss_rate_interior*100:.1f}%)")
            print(f"    Visualizations saved to: {unmatched_dir}")

            # CSV ???(boundary ?뺣낫 ?ы븿)
            csv_path = diagnostic_dir / "unmatched_gt_details.csv"
            unmatched_df.to_csv(csv_path, index=False)
            print(f"    Details saved to: {csv_path}")
        else:
            print(f"    No unmatched GT particles found!")

    def _create_diagnostic_visualizations(self, viz_dir: Path):
        """Create near-miss diagnostic visualizations."""
        diagnostic_dir = viz_dir / "diagnostics"
        diagnostic_dir.mkdir(exist_ok=True)

        near_miss_count = 0

        for img_name, data in self.visualization_data.items():
            diagnostics = data.get('diagnostics')
            if diagnostics is None or len(diagnostics['near_misses']) == 0:
                continue

            image = data['image']
            gt_masks = data['gt_masks']
            pred_masks = data['pred_masks']
            near_misses = diagnostics['near_misses']

            near_miss_count += len(near_misses)

            # RGB 蹂??
            if len(image.shape) == 2:
                rgb_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                rgb_img = image.copy()

            # Near-miss ?띿쓣 ?밸퀎???쒖떆
            for gt_idx, pred_idx, iou in near_misses:
                # GT: ?몃???
                gt_mask = gt_masks[gt_idx]
                contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rgb_img, contours, -1, (0, 255, 255), 2)  # Yellow

                # Pred: 留덉젨?
                pred_mask = pred_masks[pred_idx]
                contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rgb_img, contours, -1, (255, 0, 255), 2)  # Magenta

                # IoU 媛??쒖떆
                coords = np.argwhere(gt_mask > 0)
                if len(coords) > 0:
                    y_center = int(coords[:, 0].mean())
                    x_center = int(coords[:, 1].mean())
                    cv2.putText(rgb_img, f"IoU:{iou:.2f}", (x_center, y_center),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # 踰붾? 異붽?
            legend_h = 100
            legend = np.ones((legend_h, rgb_img.shape[1], 3), dtype=np.uint8) * 255
            cv2.putText(legend, f"Near-Misses: {len(near_misses)} pairs (0.3 <= IoU < 0.5)",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(legend, "Yellow: Unmatched GT", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(legend, "Magenta: Unmatched Pred", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            result = np.vstack([rgb_img, legend])
            cv2.imwrite(str(diagnostic_dir / f"{img_name}_near_misses.png"), result)

        if near_miss_count > 0:
            print(f"    Found {near_miss_count} near-miss cases across all images")
            print(f"    Diagnostic images saved to: {diagnostic_dir}")
        else:
            print(f"    No near-misses detected (all IoU either >= 0.5 or < 0.3)")

    def _save_complete_analysis_data(self, results_df: pd.DataFrame, stats_df: pd.DataFrame,
                                     size_bin_stats: pd.DataFrame, boundary_stats: pd.DataFrame,
                                     core_metrics: dict):
        """?꾩쟾??遺꾩꽍 ?곗씠?????(?щ텇?앹슜)

        Args:
            results_df: 留ㅼ묶???낆옄蹂??곸꽭 寃곌낵
            stats_df: ?꾩껜 ?듦퀎 ?붿빟
            size_bin_stats: ?ъ씠利?援ш컙蹂??듦퀎
            boundary_stats: 寃쎄퀎 ?④낵 遺꾩꽍
            core_metrics: 4? ?듭떖 吏??
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_data_path = self.output_dir / f"analysis_data_{timestamp}.pkl"

        # ??ν븷 ?곗씠??援ъ꽦
        complete_data = {
            # ?뚯씠?꾨씪???ㅼ젙
            'config': {
                'filter_overlapping': self.filter_overlapping,
                'dataset_dir': str(self.dataset_dir),
                'annotation_dir': str(self.annotation_dir),
                'output_dir': str(self.output_dir),
                'timestamp': timestamp
            },

            # ?듭떖 寃곌낵 ?곗씠??
            'results_df': results_df,  # 留ㅼ묶??紐⑤뱺 ?낆옄 ??(GT-Pred)
            'segmentation_stats': self.segmentation_stats,  # ?대?吏蹂?segmentation ?덉쭏
            'visualization_data': self.visualization_data,  # 留덉뒪?? 留ㅼ묶, 吏꾨떒 ?뺣낫

            # ?듦퀎 ?곗씠??
            'stats_df': stats_df,  # ?꾩껜 ?듦퀎 ?붿빟
            'size_bin_stats': size_bin_stats,  # ?ъ씠利?援ш컙蹂??듦퀎
            'boundary_stats': boundary_stats,  # 寃쎄퀎 ?④낵 遺꾩꽍

            # ?듭떖 吏??
            'core_metrics': core_metrics,  # 4? ?듭떖 吏??(recall, additional_detection_ratio, mean_iou, mape)

            # 硫뷀? ?뺣낫
            'n_images': len(self.visualization_data),
            'n_matched_pairs': len(results_df),
            'columns': {
                'results_df': list(results_df.columns),
                'stats_df': list(stats_df.columns) if stats_df is not None else [],
                'size_bin_stats': list(size_bin_stats.columns),
                'boundary_stats': list(boundary_stats.columns)
            }
        }

        # ???
        with open(analysis_data_path, 'wb') as f:
            pickle.dump(complete_data, f)

        print(f"\n[OK] Complete analysis data saved to: {analysis_data_path}")
        print(f"   - {len(results_df)} matched particle pairs")
        if self.segmentation_stats is not None:
            print(f"   - {len(self.segmentation_stats)} images with segmentation stats")
        if self.visualization_data:
            print(f"   - {len(self.visualization_data)} images with visualization data")
        print(f"   - 4 core metrics + all statistical analyses")
        print(f"   - Configuration: filter_overlapping={self.filter_overlapping}")
        print(f"\n   Use this file for re-analysis without re-running the pipeline:")
        print(f"      import pickle")
        print(f"      with open('{analysis_data_path}', 'rb') as f:")
        print(f"          data = pickle.load(f)")
        print(f"      results_df = data['results_df']")
        print(f"      core_metrics = data['core_metrics']")

    def run_full_pipeline(self, original_images_dir: str, max_images: Optional[int] = None):
        """
        ?꾩껜 ?뚯씠?꾨씪???ㅽ뻾

        Args:
            original_images_dir: ?먮낯 ?대?吏 ?대뜑 寃쎈줈
            max_images: 泥섎━??理쒕? ?대?吏 媛쒖닔 (None = ?꾩껜)
        """
        print("Starting Size Validation Pipeline...")
        print(f"Preprocessing mode: {self.preprocess_mode_tag} (noise2sr={self.use_noise2sr}, epochs={self.noise2sr_epochs})")

        # Step 1: Annotation 蹂듭궗
        image_names = self.step1_copy_matching_annotations()

        # ?대?吏 媛쒖닔 ?쒗븳
        if max_images is not None:
            print(f"\nWARNING: Limiting to first {max_images} images (out of {len(image_names)})")
            image_names = set(list(image_names)[:max_images])

        if self.reuse_preprocessed_only:
            self.verify_reusable_preprocessed_images(image_names)

        # Step 2: Annotation ?낅뜲?댄듃
        self.step2_update_annotations(image_names, Path(original_images_dir))

        # Step 3: SAM2 ?ㅽ뻾 諛?鍮꾧탳
        results_df = self.step3_run_sam2_and_compare(image_names)

        # NOTE: intermediate_results.pkl is already saved in step3 (line 788)
        # No need to save again here - it would overwrite the dict format with DataFrame only!

        # Step 4: ?듦퀎 怨꾩궛 諛????
        stats_df, excel_path = self.step4_calculate_statistics(results_df)

        # Step 5: ?쒓컖???앹꽦
        viz_dir = self.step5_visualize_results(results_df)

        print("\n=== Pipeline Complete ===")
        return results_df, stats_df, excel_path

    def resume_from_step4(self, intermediate_results_path: str):
        """
        Step 3 以묎컙 寃곌낵?먯꽌 ?ш컻 (Step 4, 5留??ㅽ뻾)

        Args:
            intermediate_results_path: Step 3 寃곌낵 pickle ?뚯씪 寃쎈줈

        Returns:
            results_df, stats_df, excel_path

        Example:
            validator = SizeValidator(...)
            results_df, stats_df, excel_path = validator.resume_from_step4(
                'workspace_outputs/validation/size_validation/intermediate_results.pkl'
            )
        """
        print("\n" + "="*80)
        print("RESUMING FROM STEP 4 (Statistics & Visualization)")
        print("="*80)

        # Load intermediate results
        print(f"\nLoading intermediate results: {intermediate_results_path}")
        import pickle
        with open(intermediate_results_path, 'rb') as f:
            intermediate_data = pickle.load(f)

        # Handle both old (DataFrame only) and new (dict) formats
        if isinstance(intermediate_data, dict):
            results_df = intermediate_data['results_df']
            self.segmentation_stats = intermediate_data['segmentation_stats']
            self.visualization_data = intermediate_data.get('visualization_data', {})
            print(f"   Loaded {len(results_df)} matched particles")
            print(f"   Loaded segmentation_stats with boundary counts")
            if self.visualization_data:
                print(f"   Loaded visualization_data for {len(self.visualization_data)} images")
        else:
            # Old format: just DataFrame
            results_df = intermediate_data
            self.segmentation_stats = None
            self.visualization_data = {}
            print(f"   Loaded {len(results_df)} matched particles (old format)")
            print(f"   WARNING: No segmentation_stats/visualization_data available")

        # Step 4: ?듦퀎 怨꾩궛 諛????
        stats_df, excel_path = self.step4_calculate_statistics(results_df)

        # Step 5: ?쒓컖???앹꽦
        viz_dir = self.step5_visualize_results(results_df)

        print("\n=== Resume Complete ===")
        return results_df, stats_df, excel_path

    def _resolve_results_excel_path(self, results_excel_path: str) -> Path:
        """Resolve existing results excel path, accepting optional .xlsx suffix."""
        if not results_excel_path:
            raise ValueError("results_excel_path is empty")

        raw = Path(results_excel_path)
        raw_with_xlsx = raw if raw.suffix.lower() == ".xlsx" else raw.with_suffix(".xlsx")

        candidates = [raw, raw_with_xlsx]
        if not raw.is_absolute():
            candidates.extend(
                [
                    self.output_dir / raw,
                    self.output_dir / raw_with_xlsx,
                    Path.cwd() / raw,
                    Path.cwd() / raw_with_xlsx,
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
        raise FileNotFoundError(
            f"Could not find results excel file.\nTried:\n{attempted}"
        )

    def resume_from_results_excel(self, results_excel_path: str, regenerate_excel: bool = False):
        """
        Resume from an existing size_validation_results excel (skip SAM/Step 3).

        Args:
            results_excel_path: Path or stem of size_validation_results_*.xlsx
            regenerate_excel: If True, rerun Step 4 and save a new excel

        Returns:
            results_df, stats_df, excel_path, viz_dir
        """
        print("\n" + "=" * 80)
        print("RESUMING FROM EXISTING RESULTS EXCEL")
        print("=" * 80)

        excel_path_resolved = self._resolve_results_excel_path(results_excel_path)
        print(f"\nLoading results excel: {excel_path_resolved}")

        xls = pd.ExcelFile(excel_path_resolved)
        target_sheet = "Detailed Results"
        if target_sheet not in xls.sheet_names:
            required_cols = {
                'image', 'gt_idx', 'iou', 'size_error_pct',
                'gt_size', 'pred_size', 'size_error', 'gt_boundary_flag'
            }
            target_sheet = None
            for sheet in xls.sheet_names:
                preview = pd.read_excel(excel_path_resolved, sheet_name=sheet, nrows=30)
                if required_cols.issubset(set(preview.columns)):
                    target_sheet = sheet
                    break
            if target_sheet is None:
                raise ValueError(
                    "No compatible sheet found in results excel. "
                    "Expected 'Detailed Results' or an equivalent sheet with matched-pair columns."
                )

        results_df = pd.read_excel(excel_path_resolved, sheet_name=target_sheet)
        print(f"   Loaded {len(results_df)} matched rows from sheet: {target_sheet}")

        if "Segmentation Quality" in xls.sheet_names:
            self.segmentation_stats = pd.read_excel(excel_path_resolved, sheet_name="Segmentation Quality")
            print(f"   Loaded Segmentation Quality: {len(self.segmentation_stats)} rows")
        else:
            self.segmentation_stats = None
            print("   Segmentation Quality sheet not found (recall/extra detection may be unavailable).")

        # Excel reuse mode does not include raw per-image masks
        self.visualization_data = {}

        if regenerate_excel:
            print("   Regenerating summary excel via Step 4...")
            stats_df, excel_path = self.step4_calculate_statistics(results_df)
            excel_path = str(excel_path) if excel_path is not None else str(excel_path_resolved)
        else:
            stats_df = None
            excel_path = str(excel_path_resolved)
            print("   Skipping Step 4 excel regeneration (using existing file).")

        print("   Generating visualizations from loaded results...")
        viz_dir = self.step5_visualize_results(results_df)

        print("\n=== Excel Resume Complete ===")
        return results_df, stats_df, excel_path, viz_dir


# ?ㅽ뻾 ?덉젣
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Size validation pipeline")
    parser.add_argument(
        "results_excel",
        nargs="?",
        default=None,
        help="Existing size_validation_results_*.xlsx (or stem without .xlsx) to reuse.",
    )
    parser.add_argument(
        "--results-excel",
        dest="results_excel_flag",
        type=str,
        default=None,
        help="Same as positional results_excel.",
    )
    parser.add_argument(
        "--regenerate-excel",
        action="store_true",
        help="When reusing --results-excel, rerun Step 4 and save a new summary excel.",
    )
    parser.add_argument(
        "--publication-metrics-only",
        action="store_true",
        help=(
            "Read Detailed Results and Segmentation Quality from an existing Excel "
            "and create publication metrics without preprocessing, SAM2, or plots."
        ),
    )
    parser.add_argument(
        "--publication-output",
        type=Path,
        default=None,
        help=(
            "Optional output path for --publication-metrics-only. Defaults to "
            "size_validation/size_validation_publication_metrics.xlsx."
        ),
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=10000,
        help="Number of image-cluster bootstrap resamples (default: 10000).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="Deterministic bootstrap random seed (default: 42).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit number of images for full pipeline mode.",
    )
    parser.add_argument(
        "--expected-images",
        type=int,
        default=None,
        help="Optional exact Dataset_size image-count guard (use 200 for the full run).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Dataset_size directory path for full pipeline mode.",
    )
    parser.add_argument(
        "--annotation-dir",
        type=str,
        default=None,
        help="Original annotation directory path (e.g., .../ds/ann).",
    )
    parser.add_argument(
        "--original-images-dir",
        type=str,
        default=None,
        help="Original full-image directory path (e.g., .../ds/img).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output root directory. Defaults to get_validation_dir().",
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
        "--reuse-preprocessed-only",
        action="store_true",
        help=(
            "Use existing verified BM3D+Noise2SR PNGs only and fail if any are "
            "missing; never execute BM3D or Noise2SR."
        ),
    )
    args = parser.parse_args()

    for name, value in (
        ("--pred-iou-thresh", args.pred_iou_thresh),
        ("--stability-score-thresh", args.stability_score_thresh),
    ):
        if not 0.0 <= value <= 1.0:
            parser.error(f"{name} must be between 0 and 1")
    if args.expected_images is not None and args.expected_images < 1:
        parser.error("--expected-images must be at least one")

    results_excel_input = args.results_excel_flag or args.results_excel

    # 寃쎈줈 ?ㅼ젙 (script ?꾩튂 湲곗??쇰줈 怨좎젙)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    default_dataset_dir = PROJECT_ROOT / "Dataset_size"
    default_annotation_dir = PROJECT_ROOT / "emps-DatasetNinja" / "ds" / "ann"
    default_original_images_dir = PROJECT_ROOT / "emps-DatasetNinja" / "ds" / "img"
    DATASET_DIR = str(Path(args.dataset_dir).resolve()) if args.dataset_dir else str(default_dataset_dir)
    ANNOTATION_DIR = (
        str(Path(args.annotation_dir).resolve()) if args.annotation_dir else str(default_annotation_dir)
    )
    ORIGINAL_IMAGES_DIR = (
        str(Path(args.original_images_dir).resolve())
        if args.original_images_dir
        else str(default_original_images_dir)
    )
    OUTPUT_DIR = str(Path(args.output_dir).resolve()) if args.output_dir else str(get_validation_dir())

    # Existing-workbook modes validate the workbook's image count downstream
    # and do not require private Dataset_size images in the public repository.
    if args.expected_images is not None and not results_excel_input:
        dataset_images = {
            path.resolve()
            for path in Path(DATASET_DIR).iterdir()
            if path.is_file()
            and path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        }
        if len(dataset_images) != args.expected_images:
            parser.error(
                f"Expected exactly {args.expected_images} Dataset_size images, "
                f"found {len(dataset_images)} in {DATASET_DIR}"
            )

    # ===== ?대?吏 媛쒖닔 ?쒗븳 ?ㅼ젙 =====
    MAX_IMAGES = None  # 理쒖쥌 寃利? ?꾩껜 ??泥섎━
    # ===================================

    # ===== Overlap ?꾪꽣留??ㅼ젙 =====
    FILTER_OVERLAPPING = True  # True: 寃뱀튂硫??묒? ?낆옄留??몄떇 (湲곕낯媛?
                                 # False: 寃뱀퀜??紐⑤몢 ?몄떇
    # ==================================

    # ===== Preprocessing ?ㅼ젙 =====
    USE_NOISE2SR = True         # Noise2SR is always ON
    NOISE2SR_EPOCHS = 1500      # None: config 湲곕낯媛??ъ슜
    # =================================

    # Validator ?앹꽦 諛??ㅽ뻾
    print("Using paths:")
    print(f"  - dataset_dir: {DATASET_DIR}")
    print(f"  - annotation_dir: {ANNOTATION_DIR}")
    print(f"  - original_images_dir: {ORIGINAL_IMAGES_DIR}")
    print(f"  - output_dir: {OUTPUT_DIR}")
    print(f"  - pred_iou_thresh: {args.pred_iou_thresh:.2f}")
    print(f"  - stability_score_thresh: {args.stability_score_thresh:.2f}")
    print(f"  - reuse_preprocessed_only: {args.reuse_preprocessed_only}")

    validator = SizeValidator(DATASET_DIR, ANNOTATION_DIR, OUTPUT_DIR,
                             filter_overlapping=FILTER_OVERLAPPING,
                             use_noise2sr=USE_NOISE2SR,
                             noise2sr_epochs=NOISE2SR_EPOCHS,
                             pred_iou_thresh=args.pred_iou_thresh,
                             stability_score_thresh=args.stability_score_thresh,
                             expected_images=args.expected_images,
                             reuse_preprocessed_only=args.reuse_preprocessed_only)
    if args.publication_metrics_only:
        if not results_excel_input:
            candidates = sorted(
                validator.output_dir.glob('size_validation_results_*.xlsx'),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if not candidates:
                parser.error(
                    "--publication-metrics-only requires results_excel when no "
                    "size_validation_results_*.xlsx exists in the output directory"
                )
            results_excel_input = str(candidates[0])
            print(f"Using latest results Excel: {results_excel_input}")
        publication_path = validator.generate_publication_metrics_from_excel(
            results_excel_path=results_excel_input,
            output_path=args.publication_output,
            n_bootstrap=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        )
        print(f"\nPublication metrics complete: {publication_path}")
    elif results_excel_input:
        results_df, stats_df, excel_path, viz_dir = validator.resume_from_results_excel(
            results_excel_input,
            regenerate_excel=args.regenerate_excel,
        )
        print(f"\nValidation complete from existing excel: {excel_path}")
        print(f"Visualization directory: {viz_dir}")
    else:
        max_images = args.max_images if args.max_images is not None else MAX_IMAGES
        results_df, stats_df, excel_path = validator.run_full_pipeline(
            ORIGINAL_IMAGES_DIR,
            max_images=max_images  # ?대?吏 媛쒖닔 ?쒗븳
        )
        print(f"\nValidation complete! Check {excel_path} for results.")
