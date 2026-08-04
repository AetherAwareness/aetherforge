"""Export helpers — HF hub push stubs and local agent directory layout."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from aetherforge.utils.logging import get_logger

log = get_logger("packaging.export")


def export_agent_directory(aetherpackage_dir: str | Path, dest: str | Path) -> Path:
    """
    Copy an AetherPackage into a flat agent-consumable directory:
      adapters/  orchestration.yaml  MODEL_CARD.md  scorecard.json
    """
    src = Path(aetherpackage_dir)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name in (
        "orchestration.yaml",
        "MODEL_CARD.md",
        "scorecard.json",
        "affinity.json",
        "manifest.json",
        "continuous_protocol.json",
        "config.resolved.yaml",
    ):
        p = src / name
        if p.exists():
            shutil.copy2(p, dest / name)
    # adapter symlink/copy if referenced
    man = src / "manifest.json"
    if man.exists():
        import json

        meta = json.loads(man.read_text())
        adapter = meta.get("files", {}).get("adapter_dir")
        if adapter and Path(adapter).exists():
            target = dest / "adapters"
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(adapter, target)
    log.info("Exported agent directory → %s", dest)
    return dest


def push_to_hub(
    local_dir: str | Path,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = True,
) -> str:
    """Optional HF Hub push (requires huggingface_hub login / token)."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=str(local_dir), repo_id=repo_id)
    log.info("Pushed %s → hub:%s", local_dir, repo_id)
    return repo_id
