"""Reproducibility helpers: content hashes for data, configs, artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def json_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dataset_fingerprint(
    records: Iterable[Mapping[str, Any]],
    text_keys: tuple[str, ...] = ("text", "prompt", "completion", "messages"),
    sample_limit: int = 50_000,
) -> str:
    """Stable fingerprint of a dataset stream (content-based, order-sensitive up to limit)."""
    h = hashlib.sha256()
    n = 0
    for rec in records:
        parts: list[str] = []
        for k in text_keys:
            if k in rec and rec[k] is not None:
                parts.append(f"{k}={rec[k]}")
        if not parts:
            parts.append(json.dumps(rec, sort_keys=True, default=str))
        h.update(("|".join(parts) + "\n").encode("utf-8"))
        n += 1
        if n >= sample_limit:
            break
    h.update(f"count={n}".encode())
    return h.hexdigest()
