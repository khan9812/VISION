# Release verification

Historical numerical and neural checks below were recorded on 2026-09-15. Publication guidance was updated on 2026-09-21.

## Completed checks

- All 47 Python source files in the original preparation parsed successfully. The current package has 46 Python files after removing the manuscript-only figure renderer.
- 14 regression tests pass: executable Noise2SR settings and JSON round trip; cache invalidation; GUI preprocessed-image data flow; PF-SUI degeneracy, sample SD and scale invariance; particle deletion and empty exports; retained morphology; regenerated spatial plots; external unavailable-PF outputs; mask crop bounds; background/nested filtering; representative-based stopping including equality.
- 56 numerical comparisons pass from the included observations: all 256 screening selections, area/IoU estimates and image-cluster CIs, shape accuracy/macro-F1 CIs, six PF-SUI condition means/CIs, 1,440 preprocessing F1 values reconstructed from counts, and final external-case counts/PF-SUI.
- Streamlit first-page AppTest: zero exceptions. CSV/XLSX/PNG/ZIP exports are exercised in regression tests; exported ZIP integrity passes.
- Six analysis CLI entry points return successful `--help` output.
- Real Noise2SR optimization runs on a synthetic 128×128 image with patch 128, batch 12, one epoch, M=1, CPU; finite output retains the input dimensions.
- Real SAM 2.1 and CLIP inference succeeds using existing local weights on a synthetic image. The SAM smoke test uses a reduced 4×4 point grid and returns zero retained masks; this checks execution and the empty-mask path, not detection quality. CLIP classifies an explicitly supplied synthetic mask.
- Manuscript figures and pre-rendered validation plots are omitted from the current code package. GUI illustration assets and all numerical inputs used by the statistical checks are retained.

## Local recheck — 2026-09-21

After renaming VISION to Observation and removing pre-rendered figures, all 46 remaining Python files parse, the 14 regression tests pass, and all 56 archived-statistics checks pass. Streamlit first-page testing reports zero exceptions and displays the new full name. These checks use the existing local environment; they do not establish a successful remote Actions run or a clean installation.

## Scope limits

Tests ran in the pre-existing Windows/Python environment listed in ENVIRONMENT.md. The release changes pandas to 2.2.3 and installs SAM from a pinned official source archive. A clean environment was created, but network installation was blocked by the execution service's automatic approval/usage limit. **Fresh installation and pip check remain unverified.** The included GitHub Actions workflow runs those checks after upload; no successful Actions run is claimed here.

GUI tests intercept neural stages when checking data flow. The separate neural smoke tests use actual networks but do not execute the full 1,500-epoch workflow or rerun the historical datasets. Archived numerical observations are preserved. Earlier benchmark Noise2SR configuration provenance is incomplete, and corrected crop/centroid handling may change fresh predictions. Statistical reproduction of saved observations must not be presented as exact neural reproduction.

Raw images, original prepared crops, mask-linked manual annotations and weights must be obtained separately for full inference. The public repository is https://github.com/khan9812/VISION. Inspection on 2026-09-21 found that its main branch lacked the workflow and analysis scripts. The revised local package must be pushed before those files are available remotely. No successful GitHub Actions run, release tag, or software DOI is claimed here.

## Repeat the local checks

```text
python -m unittest discover -s tests -v
python analysis/reproduce_results.py
```

Run with the intended environment's Python after following README installation instructions. The numerical report is written to `workspace_outputs/reproduction/numeric_verification.json`.

## GitHub Actions

`Release verification` is the workflow name declared by `.github/workflows/verification.yml`; it is not a built-in GitHub menu. Place that file at the repository root with `analysis/`, `app/`, `modules/`, `results/`, `scripts/`, `tests/`, and the requirements files. Upload the contents of the public folder, preserving dot-prefixed folders. Pushing only the YAML is insufficient when `analysis/` is missing.

1. Commit and push the complete package to `main` in https://github.com/khan9812/VISION.
2. Open https://github.com/khan9812/VISION/actions (the `/actions/new` page is the workflow creation screen).
3. Open **Release verification**, select the latest run for the pushed commit, and open the **verify** job. A push starts the supplied workflow automatically.
4. If a manual run is needed, select **Run workflow**, branch **main**, then **Run workflow**. The file must be on the default branch for this control to appear.
5. Confirm installation and `pip check`, regression tests, and archived-statistics reproduction succeed. On failure, open the red step and inspect its log; a listed workflow alone does not mean the tests passed.
6. On a successful run's summary page, download **numeric-verification** under **Artifacts**. Open `numeric_verification.json`; its summary should show 56 checks and 56 passes.

These checks do not require manuscript figure files, GPU weights, a GitHub Release, or a DOI. They verify installation, regression behavior, and recomputation of archived observations; they do not rerun the full neural benchmark.

Official instructions: https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow
