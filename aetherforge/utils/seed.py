"""Global seed control for reproducible AetherForge runs."""

from __future__ import annotations

import os
import random
from typing import Optional

from aetherforge.utils.logging import get_logger

log = get_logger("utils.seed")


def set_global_seed(seed: int, deterministic_torch: bool = False) -> None:
    """Seed Python, NumPy, and (if present) PyTorch RNGs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
    except Exception:
        pass
    log.info("Global seed set to %d", seed)


def derive_seed(base: int, *parts: str) -> int:
    """Stable derived seed for sub-stages (data, probe, train)."""
    h = base
    for p in parts:
        for ch in p:
            h = (h * 131 + ord(ch)) % (2**31 - 1)
    return int(h)
