"""Dataset storage serialization helpers (memory-mapped binary)."""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor


def save_bin(
    file_path: str,
    tensor_group: Dict[str, List[Tensor]],
    record_keys: Optional[List[str]] = None,
):
    """Save tensors as memory-mapped binary files.

    When *record_keys* is provided, those keys are written with per-record
    cumulative offsets in ``meta.json`` so that ``MmapStore.fetch_record``
    can slice individual records from the concatenated binary without
    cross-record concatenation.  Keys not in *record_keys* (e.g. SEQ
    ``sequence``) are written as a single contiguous stream without
    offsets, preserving backward compatibility.

    Nested keys (``List[List[Tensor]]`` such as GRPO ``responses``) are
    not supported in bin format — use JSONL for those.
    """
    os.makedirs(file_path, exist_ok=True)
    record_keys = set(record_keys or [])
    meta = {}
    for key, tensors in tensor_group.items():
        if tensors and isinstance(tensors[0], list):
            raise ValueError(
                f"Nested key '{key}' (List[List[Tensor]]) is not supported "
                f"in bin format. Use JSONL storage instead."
            )
        cat = torch.cat(tensors, dim=0)
        entry: Dict[str, Any] = {
            "shape": list(cat.shape),
            "dtype": str(cat.dtype).split(".")[-1],
        }
        if key in record_keys:
            offsets = [0]
            for t in tensors:
                offsets.append(offsets[-1] + t.shape[0])
            entry["offsets"] = offsets
        meta[key] = entry
        np.asarray(cat.cpu().numpy()).tofile(os.path.join(file_path, f"{key}.bin"))
    with open(os.path.join(file_path, "meta.json"), "w") as f:
        json.dump(meta, f)


def load_bin(file_path: str) -> Dict[str, List[Tensor]]:
    with open(os.path.join(file_path, "meta.json"), "r") as f:
        meta = json.load(f)
    segments: Dict[str, List[Tensor]] = {}
    for key, info in meta.items():
        arr = np.memmap(
            os.path.join(file_path, f"{key}.bin"),
            dtype=info["dtype"],
            mode="c",
            shape=tuple(info["shape"]),
        )
        segments[key] = [torch.from_numpy(arr)]
    return segments


def load_bin_offsets(file_path: str) -> Dict[str, List[int]]:
    """Read per-record cumulative offsets from ``meta.json``.

    Returns an empty dict when no key has offsets (legacy bin files),
    in which case record-mode access falls back to per-record segment
    indexing (JSONL layout).
    """
    with open(os.path.join(file_path, "meta.json"), "r") as f:
        meta = json.load(f)
    offsets: Dict[str, List[int]] = {}
    for key, info in meta.items():
        if "offsets" in info:
            offsets[key] = info["offsets"]
    return offsets
