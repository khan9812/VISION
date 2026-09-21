"""Lossless compressed storage for post-processed SAM parameter masks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


PARAMETER_MASK_CACHE_VERSION = "binary_rle_npz_v1"


def _metadata_path(archive_path) -> Path:
    return Path(archive_path).with_suffix(".json")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Encode a binary mask as zero-first, row-major run lengths."""
    flat = np.ascontiguousarray(mask, dtype=bool).reshape(-1)
    if flat.size == 0:
        return np.asarray([0], dtype=np.uint64)
    change_points = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    boundaries = np.concatenate(
        (np.asarray([0]), change_points, np.asarray([flat.size]))
    )
    runs = np.diff(boundaries).astype(np.uint64, copy=False)
    if flat[0]:
        runs = np.concatenate((np.asarray([0], dtype=np.uint64), runs))
    return runs


def decode_binary_mask(runs: np.ndarray, shape) -> np.ndarray:
    """Decode a zero-first run-length array into an exact boolean mask."""
    shape = tuple(int(value) for value in shape)
    pixel_count = int(np.prod(shape, dtype=np.int64))
    runs = np.asarray(runs, dtype=np.uint64).reshape(-1)
    if int(runs.sum(dtype=np.uint64)) != pixel_count:
        raise ValueError(
            f"RLE length {int(runs.sum())} does not match mask shape {shape} "
            f"({pixel_count} pixels)"
        )
    flat = np.zeros(pixel_count, dtype=bool)
    cursor = 0
    for run_index, run_length in enumerate(runs):
        next_cursor = cursor + int(run_length)
        if run_index % 2 == 1:
            flat[cursor:next_cursor] = True
        cursor = next_cursor
    return flat.reshape(shape)


def _encode_mask_list(masks, mask_shape):
    mask_shape = tuple(int(value) for value in mask_shape)
    encoded = []
    offsets = [0]
    for mask_record in masks:
        mask = (
            mask_record.get("segmentation")
            if isinstance(mask_record, dict)
            else mask_record
        )
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != mask_shape:
            raise ValueError(f"Mask shape {mask.shape} != expected {mask_shape}")
        runs = encode_binary_mask(mask)
        encoded.append(runs)
        offsets.append(offsets[-1] + len(runs))
    counts = (
        np.concatenate(encoded).astype(np.uint64, copy=False)
        if encoded
        else np.empty(0, dtype=np.uint64)
    )
    return counts, np.asarray(offsets, dtype=np.uint64)


def save_parameter_mask_cache(archive_path, pair_entries, metadata):
    """Save every pair's post-processed masks in a checksummed NPZ archive."""
    archive_path = Path(archive_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    mask_shape = tuple(int(value) for value in metadata["mask_shape"])

    arrays = {}
    pair_metadata = []
    for pair_index, entry in enumerate(pair_entries):
        counts, offsets = _encode_mask_list(entry.get("masks", []), mask_shape)
        key = f"pair_{pair_index:03d}"
        arrays[f"{key}_counts"] = counts
        arrays[f"{key}_offsets"] = offsets
        pair_metadata.append(
            {
                "pair_index": int(pair_index),
                "stage": int(entry["stage"]),
                "combo_idx": int(entry["combo_idx"]),
                "pred_iou": float(entry["pred_iou"]),
                "stability": float(entry["stability"]),
                "mask_count": int(len(offsets) - 1),
                "array_key": key,
            }
        )

    temp_archive = archive_path.with_name(f".{archive_path.name}.tmp")
    temp_metadata = _metadata_path(archive_path).with_name(
        f".{_metadata_path(archive_path).name}.tmp"
    )
    try:
        with temp_archive.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        archive_stat = temp_archive.stat()
        cache_metadata = dict(metadata)
        cache_metadata.update(
            {
                "cache_version": PARAMETER_MASK_CACHE_VERSION,
                "mask_shape": list(mask_shape),
                "pair_count": len(pair_metadata),
                "pairs": pair_metadata,
                "archive_size_bytes": int(archive_stat.st_size),
                "archive_sha256": _file_sha256(temp_archive),
            }
        )
        temp_metadata.write_text(
            json.dumps(cache_metadata, indent=2), encoding="utf-8"
        )
        os.replace(temp_archive, archive_path)
        os.replace(temp_metadata, _metadata_path(archive_path))
    finally:
        if temp_archive.exists():
            temp_archive.unlink()
        if temp_metadata.exists():
            temp_metadata.unlink()
    return cache_metadata


def load_parameter_mask_metadata(archive_path, verify_checksum=False):
    """Load and validate an archive's JSON sidecar."""
    archive_path = Path(archive_path)
    metadata_path = _metadata_path(archive_path)
    if not archive_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Incomplete parameter mask cache: {archive_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("cache_version") != PARAMETER_MASK_CACHE_VERSION:
        raise ValueError(
            f"Unsupported parameter mask cache version: {metadata.get('cache_version')}"
        )
    if metadata.get("archive_size_bytes") != archive_path.stat().st_size:
        raise ValueError(f"Parameter mask cache size mismatch: {archive_path}")
    if verify_checksum and metadata.get("archive_sha256") != _file_sha256(archive_path):
        raise ValueError(f"Parameter mask cache checksum mismatch: {archive_path}")
    return metadata


def load_parameter_pair_masks(archive_path, pair_index, metadata=None):
    """Decode one parameter pair without loading masks from other pairs."""
    archive_path = Path(archive_path)
    metadata = metadata or load_parameter_mask_metadata(
        archive_path, verify_checksum=True
    )
    pair_index = int(pair_index)
    pair_metadata = metadata["pairs"][pair_index]
    if int(pair_metadata["pair_index"]) != pair_index:
        raise ValueError("Parameter mask cache pair ordering is inconsistent")
    key = pair_metadata["array_key"]
    with np.load(archive_path, allow_pickle=False) as archive:
        counts = np.asarray(archive[f"{key}_counts"], dtype=np.uint64)
        offsets = np.asarray(archive[f"{key}_offsets"], dtype=np.uint64)
    if len(offsets) != int(pair_metadata["mask_count"]) + 1:
        raise ValueError(f"Invalid mask offsets for pair {pair_index}")
    shape = tuple(int(value) for value in metadata["mask_shape"])
    return [
        decode_binary_mask(counts[int(offsets[i]) : int(offsets[i + 1])], shape)
        for i in range(len(offsets) - 1)
    ]


def load_all_parameter_masks(archive_path, metadata=None):
    """Decode all parameter pairs in schedule order."""
    archive_path = Path(archive_path)
    metadata = metadata or load_parameter_mask_metadata(
        archive_path, verify_checksum=True
    )
    shape = tuple(int(value) for value in metadata["mask_shape"])
    decoded_pairs = []
    with np.load(archive_path, allow_pickle=False) as archive:
        for expected_index, pair_metadata in enumerate(metadata["pairs"]):
            if int(pair_metadata["pair_index"]) != expected_index:
                raise ValueError("Parameter mask cache pair ordering is inconsistent")
            key = pair_metadata["array_key"]
            counts = np.asarray(archive[f"{key}_counts"], dtype=np.uint64)
            offsets = np.asarray(archive[f"{key}_offsets"], dtype=np.uint64)
            if len(offsets) != int(pair_metadata["mask_count"]) + 1:
                raise ValueError(f"Invalid mask offsets for pair {expected_index}")
            masks = [
                decode_binary_mask(
                    counts[int(offsets[i]) : int(offsets[i + 1])], shape
                )
                for i in range(len(offsets) - 1)
            ]
            decoded_pairs.append({**pair_metadata, "masks": masks})
    return decoded_pairs
