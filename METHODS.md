# Executable method definitions

## Preprocessing and masks

`modules/runtime_config.py` resolves YAML defaults, explicit JSON overrides, and an explicit epoch argument, in that order. The final pipeline uses BM3D then Noise2SR, without CLAHE. Source/configuration hashes and full settings enter the pipeline cache key. Package versions and weight-file size/modification time are recorded; the latter is a file identity check, not a cryptographic weight checksum.

`modules/sam2_utils.py` removes a background-like mask if its bounding-box area `(width × height)/(image width × image height)` exceeds 0.90, or its box reaches all four borders within 5 pixels. This uses bounding-box coverage, not foreground-mask area. Nested filtering uses the largest external contour's moment centroid: if one mask contains the other mask's centroid, the larger mask is removed. Equal-area ties follow input order. The release GUI, validation and external runner share this function.

The optional GUI area filter is disabled in `publication_final.json`. When enabled, it retains `min_particle_area ≤ area ≤ max_particle_area` by foreground-pixel counts (defaults 10 and 100000). The separate external GT rule discards reference connected components below 10 pixels; it is not a prediction cutoff.

CLIP uses the final preprocessed image, masks surroundings to black, obtains the bounding box from the segmentation mask including the last row/column, adds 15% padding, centers the crop on a square canvas, and applies CLIP preprocessing. The main labels are Circle, Triangle, Quadrilateral, Hexagon and Irregular. External cases have explicitly recorded individual/aggregate labels.

## Screening

The 13-stage schedule contains 19 pairs. Each stage representative minimizes **total** `q=FP/TP`; ties prefer more TP, fewer FP, then schedule order. TP=0 gives q=∞. Compare successive representatives. With ΔTP>0, stop at the first `R=ΔFP/ΔTP ≥ 0.5`; with ΔTP≤0, stop only if ΔFP>0. Otherwise continue.

Return the **preceding stage**. If it has two candidates, choose the smaller incremental R relative to the representative of its own preceding stage, with q-based tie breaking. Within this pair-selection step, ΔTP≤0 gives R=∞. With no crossing, select the global minimum q. The implementation is `find_optimal_stage`, `_combo_quality_key` and `_select_from_stage` in `analysis/sam_param_optimizer.py`.

## Estimands and intervals

| Quantity | Estimator and interval |
|---|---|
| Preprocessing F1/precision/recall/FP per image | Equal-weight mean across 360 images; two-sided Student-t interval |
| Preprocessing ΔF1 | Mean within-image paired difference; Student-t interval on 360 differences |
| Pooled mean IoU and area MAE | Sum across 3,932 matched pairs divided by pair count; resample 200 image clusters, retain all their pairs, recompute the pooled ratio |
| Shape accuracy | Correct predictions / classifiable particles; resample 100 image clusters and sum their confusion matrices |
| Shape macro-F1 | Equal-weight mean F1 of five fixed classes from the pooled confusion matrix; zero-denominator class F1=0; image-cluster bootstrap |
| Synthetic PF-SUI condition mean | Mean over 90 realization records per condition; percentile bootstrap within condition, seeds 42–47 |

Bootstrap CIs are the 2.5th/97.5th percentiles of 10,000 resamples. Area and shape use NumPy `default_rng(42)`. IoU/MAE are pooled estimators even though their resampling units are images. `analysis/reproduce_results.py` contains the executable calculations.

## PF-SUI

`extract_boundary_pixels_dict` obtains sites from OpenCV `RETR_EXTERNAL` / `CHAIN_APPROX_SIMPLE` contours. Distance is measured to retained contour points: a discrete approximation, not exact distance to every point of a continuous boundary segment.

The analysis domain is the convex hull of particle centroids. Voronoi cells from contour sites are clipped to this hull and combined by particle. Only particles strictly inside the hull buffered inward by 5 pixels contribute to PF-SUI; boundary particles still supply competing sites. PF-SUI is `1/(1+s_A/mean(A))`, with sample SD (`ddof=1`). Invalid hulls or fewer than two eligible regions give an unavailable result with a reason. The GUI can still report area and morphology.

Synthetic paired inputs share designed positions, but masks, centroids, hulls and eligible regions are recomputed after segmentation. Eligible-region counts differ in 2/90 C–D and 15/90 E–F pairs in the archived observations; final analysis domains are not invariably fixed.

## Historical observations and release corrections

The 403 shape observations comprise 359 transferred valid labels and 44 additional manually reviewed labels. Not all 403 are IoU≥0.5 transfers from earlier masks. The historical external runner used different centroid/tie handling; older shape validation used intensity-defined crops. The release shares these operations. These corrections can affect fresh predictions; archived observations were preserved, not silently reclassified.

The final four external runs explicitly recorded patch 128, batch 12, 1,500 epochs and four data-loader workers. Earlier benchmark caches do not establish every Noise2SR hyperparameter. The current defaults do not retrospectively prove those settings or exact neural reproducibility.

Fig. 12 retains display ranges of 0–35000 px² for projected area and 0–200000 px² for Voronoi area. Larger observations are omitted from the graphical display only; statistics use all eligible data. Omitted counts for 006/026/066/076 are 0/1/0/2 (projected area) and 1/1/0/0 (Voronoi area).
