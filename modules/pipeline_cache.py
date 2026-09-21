"""
Lightweight per-code, per-image cache utilities.

Cache layout:
    <root_dir>/<code_name>/<image_stem>/
        pipeline_cache.pkl
        pipeline_cache_meta.json
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from modules.runtime_config import runtime_signature


CACHE_SCHEMA_VERSION = 2


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def _build_signature(image_path: str, options: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], str]:
    image_file = Path(image_path)
    image_info: Dict[str, Any] = {
        "path": str(image_file.resolve()) if image_file.exists() else str(image_file),
        "exists": image_file.exists(),
    }
    if image_file.exists():
        stat = image_file.stat()
        image_info.update(
            {
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )

    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "runtime": runtime_signature(),
        "image": image_info,
        "options": options or {},
    }
    signature_text = json.dumps(payload, sort_keys=True, default=_json_default, ensure_ascii=True)
    signature_hash = hashlib.sha256(signature_text.encode("utf-8")).hexdigest()
    return payload, signature_hash


def get_pipeline_cache_paths(code_name: str, image_path: str, root_dir: str = "cache") -> Tuple[Path, Path]:
    image_stem = Path(image_path).stem
    cache_dir = Path(root_dir) / code_name / image_stem
    data_path = cache_dir / "pipeline_cache.pkl"
    meta_path = cache_dir / "pipeline_cache_meta.json"
    return data_path, meta_path


def pipeline_cache_exists(code_name: str, image_path: str, root_dir: str = "cache") -> bool:
    data_path, meta_path = get_pipeline_cache_paths(code_name, image_path, root_dir=root_dir)
    return data_path.exists() and meta_path.exists()


def load_pipeline_cache(
    code_name: str,
    image_path: str,
    options: Optional[Dict[str, Any]] = None,
    root_dir: str = "cache",
    ignore_signature: bool = False,
) -> Optional[Dict[str, Any]]:
    data_path, meta_path = get_pipeline_cache_paths(code_name, image_path, root_dir=root_dir)
    if not data_path.exists() or not meta_path.exists():
        return None

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    _, expected_hash = _build_signature(image_path, options)
    if not ignore_signature and meta.get("signature_hash") != expected_hash:
        return None

    try:
        with open(data_path, "rb") as f:
            blob = pickle.load(f)
    except (OSError, pickle.UnpicklingError):
        return None

    if isinstance(blob, dict) and "payload" in blob:
        return blob["payload"]
    if isinstance(blob, dict):
        return blob
    return None


def save_pipeline_cache(
    code_name: str,
    image_path: str,
    payload: Dict[str, Any],
    options: Optional[Dict[str, Any]] = None,
    root_dir: str = "cache",
) -> Tuple[Path, Path]:
    data_path, meta_path = get_pipeline_cache_paths(code_name, image_path, root_dir=root_dir)
    data_path.parent.mkdir(parents=True, exist_ok=True)

    signature_payload, signature_hash = _build_signature(image_path, options)

    temp_data = data_path.with_suffix(".tmp.pkl")
    with open(temp_data, "wb") as f:
        pickle.dump({"payload": payload}, f, protocol=pickle.HIGHEST_PROTOCOL)
    temp_data.replace(data_path)

    meta = {
        "code_name": code_name,
        "image_path": str(Path(image_path)),
        "signature_hash": signature_hash,
        "signature_payload": signature_payload,
        "data_file": data_path.name,
    }
    temp_meta = meta_path.with_suffix(".tmp.json")
    with open(temp_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    temp_meta.replace(meta_path)

    return data_path, meta_path
