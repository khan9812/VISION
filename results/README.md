# Final released results

Final aggregate tables and per-image/per-particle numerical observations are included here. Pre-rendered manuscript and validation plots are omitted from this code package; the numerical inputs remain available for statistical verification. Per-image masks, overlays, preprocessing outputs, caches, and pickle files are excluded.

## Data counts

| Analysis | Count |
|---|---:|
| SAM empirical screening | 256 images; size/shape validation images excluded |
| Preprocessing comparison | 360 images |
| Size validation | 200 images; 5,793 GT particles |
| Shape validation | 100 images; 403 classifiable particles |
| PF-SUI synthetic validation | 540 observations |

## SAM2 empirical-default screening

The 256-image screening set excluded the 209 unique images used in projected-area or projected-morphology validation. Predictions were matched to DatasetNinja reference masks by Hungarian assignment at mask IoU ≥ 0.5. With `R = ΔFP/ΔTP` and the engineering threshold `R = 0.5`, `(pred_iou_thresh, stability_score_thresh) = (0.95, 0.80)` was selected for 140/256 images (54.6875%) and adopted as the adjustable GUI default. Screening used BM3D only (`sigma_psd = 40/255`); user-image inference uses the selected BM3D → Noise2SR workflow and does not perform GT matching.

## Preprocessing comparison

| Metric | BM3D + Noise2SR |
|---|---:|
| Macro precision | 0.8179 (95% CI 0.7872-0.8485) |
| Macro recall | 0.8062 (95% CI 0.7745-0.8379) |
| Macro F1 | 0.7886 (95% CI 0.7576-0.8195) |
| Unmatched predictions/image | 1.497 (95% CI 1.121-1.874) |
| Paired delta F1 vs BM3D | +0.0115 (95% CI +0.0016 to +0.0214) |

## Size validation

| Metric | Value |
|---|---:|
| Predictions / matched | 4,100 / 3,932 |
| Pooled particle precision | 0.9590 (95% CI 0.9466-0.9696) |
| Image-macro recall | 0.8478 (95% CI 0.8099-0.8832) |
| Mean matched IoU | 0.8781 (95% CI 0.8658-0.8903) |
| Image-macro mean IoU | 0.9007 (95% CI 0.8927-0.9085) |
| Area MAE | 254.46 px^2 (95% CI 216.68-300.71) |
| Area MAPE | 11.07% (95% CI 9.83-12.31) |
| Signed area bias | -247.62 px^2 (95% CI -294.01 to -209.63) |
| Identity-line R^2 | 0.9960 |
| Lin's CCC | 0.9979 |

## Shape validation

Old and new masks were aligned with mask-IoU Hungarian assignment. There were 607 matched pairs and 359 transferred valid labels. After additional manual review, 403 classifiable particles were included in the final evaluation.

| Metric | Value |
|---|---:|
| Accuracy | 0.8660 (95% CI 0.7984-0.9194) |
| Macro F1 | 0.6589 (95% CI 0.5003-0.7739) |
| Weighted F1 | 0.8651 |
| Macro precision | 0.7045 |
| Macro recall / balanced accuracy | 0.6482 |

## PF-SUI

| Case | Mean | 95% CI |
|---|---:|---:|
| A | 0.995882 | 0.995002-0.996720 |
| B | 0.973150 | 0.972198-0.974067 |
| C | 0.914987 | 0.908644-0.921583 |
| D | 0.911969 | 0.905654-0.918230 |
| E | 0.763400 | 0.753708-0.773003 |
| F | 0.759693 | 0.749972-0.769512 |

## Included workbooks

- `sam_screening/optimization_results.xlsx`: final 256-image DatasetNinja GT/Hungarian screening results (mask IoU ≥ 0.5; `R = ΔFP/ΔTP`, threshold 0.5).
- `preprocessing/preprocessing_metrics.xlsx`: final four-method preprocessing comparison values.
- `size_validation/size_validation_results.xlsx`: final per-image and per-particle size results.
- `size_validation/size_validation_metrics.xlsx`: publication metrics and confidence intervals.
- `shape_validation/shape_validation_results.xlsx`: final manually reviewed shape evaluation.
- `pf_sui/pf_sui_validation_results.xlsx`: final 540-observation PF-SUI validation.

## Reproduction and final external cases

Run `python analysis/reproduce_results.py` from the repository root. It recomputes the archived estimands and CIs and checks every preprocessing F1 against its TP/FP/FN counts. Input records and cohort membership are listed in [DATA.md](../DATA.md). Preprocessing intervals are Student-t intervals; area and morphology use image-cluster bootstrap intervals, as defined in [METHODS.md](../METHODS.md).

| External case | Retained masks | Individual / Cluster | PF-eligible regions | PF-SUI |
|---|---:|---:|---:|---:|
| 006 | 10 | 9 / 1 | 4 | 0.557299 |
| 026 | 32 | 32 / 0 | 24 | 0.627459 |
| 066 | 8 | 8 / 0 | 2 | 0.879505 |
| 076 | 53 | 45 / 8 | 44 | 0.648440 |

`external_case_study/historical_run_manifest.json` records the final 2026-09-10 runs (SAM 0.95/0.80, Noise2SR patch 128/batch 12). Paths are normalized for publication. Corrected runtime code does not retroactively change these observations. The full neural benchmark was not rerun during release preparation.
