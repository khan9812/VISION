"""
Project-wide output path layout.

All generated artifacts should be stored under one root folder:
    <project_root>/workspace_outputs/

Subfolders:
    - analysis
    - optimization
    - validation
    - results
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_OUTPUT_ROOT = PROJECT_ROOT / "workspace_outputs"

ANALYSIS_DIR = WORKSPACE_OUTPUT_ROOT / "analysis"
OPTIMIZATION_DIR = WORKSPACE_OUTPUT_ROOT / "optimization"
VALIDATION_DIR = WORKSPACE_OUTPUT_ROOT / "validation"
RESULTS_DIR = WORKSPACE_OUTPUT_ROOT / "results"


def ensure_dir(path: Path) -> Path:
    """Create directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_workspace_layout() -> None:
    """Create the standard output folder layout."""
    for path in (WORKSPACE_OUTPUT_ROOT, ANALYSIS_DIR, OPTIMIZATION_DIR, VALIDATION_DIR, RESULTS_DIR):
        ensure_dir(path)


def _join_and_ensure(base: Path, *parts: str) -> Path:
    path = base.joinpath(*parts) if parts else base
    return ensure_dir(path)


def get_analysis_dir(*parts: str) -> Path:
    ensure_workspace_layout()
    return _join_and_ensure(ANALYSIS_DIR, *parts)


def get_optimization_dir(*parts: str) -> Path:
    ensure_workspace_layout()
    return _join_and_ensure(OPTIMIZATION_DIR, *parts)


def get_validation_dir(*parts: str) -> Path:
    ensure_workspace_layout()
    return _join_and_ensure(VALIDATION_DIR, *parts)


def get_results_dir(*parts: str) -> Path:
    ensure_workspace_layout()
    return _join_and_ensure(RESULTS_DIR, *parts)


def find_existing_path(preferred: Path, legacy_candidates: Optional[Iterable[Path]] = None) -> Optional[Path]:
    """
    Return the first existing path from preferred + legacy candidates.
    """
    if preferred.exists():
        return preferred
    for candidate in legacy_candidates or ():
        if candidate.exists():
            return candidate
    return None

