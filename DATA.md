# Data layout and reproduction scope

## Included observations

`results/cohort_manifest.csv` identifies each image's membership in screening, preprocessing, area and morphology cohorts. The screening set excludes the union of the 200 area and 100 morphology images (209 unique validation IDs).

- `results/preprocessing/per_image_metrics.csv`: 360 images × four methods, with full-precision precision/recall/F1 and TP/FP/FN counts.
- `results/preprocessing/paired_f1_360_images.csv`: paired BM3D and BM3D+Noise2SR F1 records.
- `results/sam_screening/optimization_results.xlsx`: all 19 candidates and selected pairs for 256 images.
- `results/size_validation/size_validation_metrics.xlsx`: matched-particle observations and per-image coverage.
- `results/shape_validation/shape_validation_results.xlsx`: 403 classified particles across 100 images.
- `results/pf_sui/pf_sui_validation_results.xlsx`: 540 realization records.
- `results/external_case_study/`: final four-image tables, Voronoi-area observations, display-omission counts, and historical run metadata.

These files are sufficient for the supplied statistical reproduction command. They do not contain raw microscopy images, manually prepared full-resolution crops, annotation bitmaps, or model weights.

## Full image/model reruns

Obtain the original EMPS images and annotations from [Yildirim and Cole's EMPS repository](https://github.com/by256/emps), preserving image IDs ([paper](https://doi.org/10.1021/acs.jcim.0c01455)). The [DatasetNinja conversion](https://datasetninja.com/emps) uses JSON objects with `objects`, `classTitle`, and `bitmap` fields (`data` and `origin`); decoding and crop alignment are implemented in `modules/reference_segmentation.py`. Full-resolution source images and crop images must correspond exactly. Crop preparation removed panel labels, annotations and scale bars; the manually chosen original crop rectangles are not reconstructed by cohort IDs alone. Exact historic image-level replay therefore also needs the authors' prepared crops.

Expected default layout for validation scripts:

```text
Dataset/                         prepared EMPS crops, named by image ID
Dataset_size/                    the 200 area-validation crops
Dataset_shape/                   the 100 morphology-validation crops
emps-DatasetNinja/ds/img/         full-size source images with matching IDs
emps-DatasetNinja/ds/ann/         <image_id>.<extension>.json (or <image_id>.json)
checkpoints/sam2.1_hiera_large.pt
```

Screening additionally uses the central 80% region of the prepared input (`_central_region_box` in `sam_param_optimizer.py`); reference masks receive the same crop. CLI options override dataset and annotation locations. The shape-validation workflow also requires the reviewed class annotations; the released per-particle labels are sufficient for aggregate statistics, not a replacement for mask-linked annotations in a fresh detection run. Consult each script's `--help` before a full rerun and use a new output directory for new settings.

For external cases, prepare:

```text
data/case_study/
  006.tif    006_gt.tif
  026.tif    026_gt.tif
  066.tif    066_gt.tif
  076.tif    076_gt.tif
```

Each `_gt.tif` is a binary reference image aligned to its input. Components below 10 pixels are ignored in GT matching. The external source is [Treder et al., nNPipe (2023)](https://doi.org/10.1038/s41524-022-00949-7), whose data-availability statement links the [source data archive](https://zenodo.org/records/7024893). Use the authors' exact selected crops and masks for historical replay. The runner uses 0.95/0.80, Noise2SR 128/12, four persistent training workers and generator seed 42 by default; it validates the layout with `--dry-run`.

Source images, prepared crops and full annotation masks are not redistributed in this release. A data-availability statement should describe this scope. If these assets are later deposited separately, add their persistent location, version, terms and crop-coordinate manifest here.
