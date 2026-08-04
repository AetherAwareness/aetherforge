"""Curated corpus loaders — domain corpus ingestion helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

from aetherforge.utils.logging import get_logger

log = get_logger("data.curators")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_chat_export(path: str | Path) -> list[dict[str, Any]]:
    """Load OpenAI-style chat JSONL ({messages:[...]}) into train records."""
    recs = load_jsonl(path)
    normalized = []
    for r in recs:
        if "messages" in r:
            normalized.append(r)
        elif "text" in r:
            normalized.append(r)
        else:
            normalized.append({"text": json.dumps(r), "source": "chat_export_raw"})
    log.info("Loaded %d chat records from %s", len(normalized), path)
    return normalized


def merge_corpora(*corpora: Iterable[dict[str, Any]], tag: Optional[str] = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for corp in corpora:
        for r in corp:
            if tag:
                r = {**r, "corpus_tag": tag}
            out.append(r)
    return out


def filter_by_source(records: list[dict[str, Any]], sources: set[str]) -> list[dict[str, Any]]:
    return [r for r in records if r.get("source") in sources]
