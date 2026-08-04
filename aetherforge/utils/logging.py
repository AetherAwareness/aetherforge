"""Structured logging for AetherForge runs."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_CONFIGURED = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str | Path] = None,
    run_id: Optional[str] = None,
) -> None:
    global _CONFIGURED
    root = logging.getLogger("aetherforge")
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    if run_id:
        root = logging.LoggerAdapter(root, {"run_id": run_id})  # type: ignore[assignment]

    _CONFIGURED = True


def get_logger(name: str = "aetherforge") -> logging.Logger:
    if not _CONFIGURED:
        setup_logging()
    if not name.startswith("aetherforge"):
        name = f"aetherforge.{name}"
    return logging.getLogger(name)
