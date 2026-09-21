# VISION

**Versatile Intelligent Segmentation for Image-based Observation of Nanoparticles**

Repository: https://github.com/khan9812/VISION

VISION measures projected particle area, classifies projected morphology, and evaluates spatial uniformity in electron microscopy images. It combines BM3D → Noise2SR, SAM 2.1, CLIP, and the particle-footprint spatial uniformity index (PF-SUI). The interface supports ROI selection, manual scale calibration, particle deletion, and CSV/XLSX/PNG/ZIP export.

This package contains the application, executable settings, analysis scripts, and numerical observations underlying the archived statistics. See [results](results/README.md), [method definitions](METHODS.md), [data layout](DATA.md), and [verification scope](RELEASE_VERIFICATION.md).

## Install and launch

Use Python 3.11 in a new environment. From this repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/install_sam2.py
.\.venv\Scripts\python.exe -m pip check
```

For NVIDIA CUDA 12.1, install the corresponding PyTorch build **before** requirements.txt:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

On Linux/macOS, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`. CPU execution is supported, but per-image training can be slow. The SAM installer uses a pinned official source archive and disables the optional compiled CUDA extension. PyTorch GPU inference remains available with a CUDA-enabled build. See [ENVIRONMENT.md](ENVIRONMENT.md).

Download `sam2.1_hiera_large.pt` from [official SAM 2](https://github.com/facebookresearch/sam2#download-checkpoints) into `checkpoints/sam2.1_hiera_large.pt`. CLIP downloads weights on first use to the user's `.cache/clip` directory. Weights are not bundled.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

Upload an image, select its ROI, enter scale calibration if available, choose analyses, and run. Without calibration, areas use pixels squared. JSON settings and exports record the full Noise2SR configuration and provenance. Noise2SR requires an ROI of at least 64 pixels along both dimensions.

## Manuscript settings

| Component | Setting |
|---|---|
| SAM | SAM 2.1 Hiera Large; predicted IoU 0.95; stability 0.80 |
| Mask generator | points/side 32; points/batch 256; crop layers 1; downscale 2; box/crop NMS 0.7; m2m enabled |
| BM3D | sigma 40 on 0–255 scale; 40/255 for normalized input |
| Noise2SR | patch 128; batch 12; 1,500 epochs; Adam lr 0.0001; stride 2; M=50; seed 42 |
| Data loader | GUI: 0 workers; final external cases: 4 persistent workers, generator seed 42 |
| CLIP | ViT-L/14@336px; batch 64; learned logit scale; highest-probability label |
| CLIP input | Segmentation-mask crops of the final preprocessed image |
| Optional particle-area filter | Disabled; stored minimum of 10 pixels is inactive |
| PF-SUI | Retained contour-point sites; convex-hull clipping; 5-pixel inward centroid exclusion; sample SD |

Load `configs/publication_final.json` in the GUI. `configs/external_case_study.json` records the external settings. Smaller images reduce the training patch to an even size. Historical benchmark caches do not establish every past Noise2SR hyperparameter; current defaults are not retrospective proof.

Offline screening used BM3D only on 256 images, excluding the area/morphology validation union. Each stage representative minimizes total FP/TP. The first representative-to-representative `R=ΔFP/ΔTP ≥ 0.5` returns the preceding stage; its pair minimizes incremental R from the earlier representative. The modal pair (0.95, 0.80) occurred in 140/256 images. User-image inference does not use GT matching. See [METHODS.md](METHODS.md).

## Statistical reproduction

Use the environment's Python executable. No model weights or source images are needed:

```powershell
python analysis/reproduce_results.py
python -m unittest discover -s tests -v
```

The first command writes `workspace_outputs/reproduction/numeric_verification.json` and exits nonzero on mismatch. It checks the 256 screening selections, pooled IoU/area and shape bootstrap CIs, PF-SUI means/CIs, and 1,440 preprocessing records against TP/FP/FN.

For fresh external-case inference after preparing [DATA.md](DATA.md):

```powershell
python analysis/run_case_study_batch.py --case-study-dir data/case_study --output-dir workspace_outputs/external_cases --dry-run
python analysis/run_case_study_batch.py --case-study-dir data/case_study --output-dir workspace_outputs/external_cases
```

Other entry points under `analysis/` are `sam_param_optimizer.py`, `compare_preprocess_full.py`, `size_validation.py`, `shape_validation.py`, and `distribution_validation.py`; each supports `--help`. Full neural reproduction needs original crops and annotations supplied separately. Archived statistics were preserved; the complete neural benchmark was not rerun for this release.

## Repository and reuse

`app/`: GUI; `modules/`: shared algorithms; `analysis/`: validation/reproduction; `configs/`: settings; `results/`: observations and summaries; `tests/`: regression checks. Runtime caches and datasets are ignored by Git.

VISION-authored code retains its existing MIT license. Third-party software, adapted Noise2SR, weights and source images retain their own terms: see [THIRD_PARTY.md](THIRD_PARTY.md). For publications, record the repository URL and the release tag or commit used. Archiving a release with a software DOI is optional. See [verification instructions](RELEASE_VERIFICATION.md#github-actions).
