"""
Multi-theme affinity probes.

Offline: keyword overlap of the domain pack against the forensic theme bank.
Live: run AffinityProbe once per theme when a model bundle is loaded.

Scores feed forensic content labels — they never invent industry tables.
"""

from __future__ import annotations

from typing import Any, Optional

from aetherforge.utils.logging import get_logger

log = get_logger("affinity.theme_probe")


def collect_theme_probes(
    pack: Any = None,
    *,
    per_theme: int = 3,
) -> list[dict[str, str]]:
    from aetherforge.groups.forensics import probe_texts_from_theme_bank

    items = probe_texts_from_theme_bank(per_theme=per_theme)
    if pack is not None:
        domain = getattr(pack, "domain", "") or ""
        for i, topic in enumerate(list(getattr(pack, "topics", []) or [])[:8]):
            items.append(
                {
                    "theme_id": f"pack:{domain}",
                    "theme_label": domain or "pack",
                    "text": topic,
                    "probe_id": f"pack:{domain}:{i}",
                }
            )
    return items


def score_themes_offline(
    domain: str,
    keywords: Optional[list[str]] = None,
    pack: Any = None,
) -> dict[str, float]:
    """Field-agnostic overlap of domain/pack tokens vs the theme bank."""
    from aetherforge.groups.forensics import DEFAULT_THEME_BANK

    kws = [k.lower() for k in (keywords or [])]
    if pack is not None:
        kws.extend(str(k).lower() for k in (getattr(pack, "keywords", None) or []))
        kws.extend(str(t).lower() for t in (getattr(pack, "topics", None) or [])[:8])
    hay = " ".join([domain.lower(), *kws])
    scores: dict[str, float] = {}
    for tid, spec in DEFAULT_THEME_BANK.items():
        bank = [str(w).lower() for w in (spec.get("keywords") or [])]
        if not bank:
            scores[tid] = 0.0
            continue
        hits = sum(1 for w in bank if w in hay)
        scores[tid] = hits / len(bank)
    return scores


def attach_theme_scores(
    affinity: Any,
    *,
    pack: Any = None,
    live_scores: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Write theme scores onto affinity.metadata (mutates). Returns the payload."""
    meta = dict(getattr(affinity, "metadata", None) or {})
    offline = score_themes_offline(
        getattr(affinity, "domain", "") or "",
        pack=pack,
    )
    payload = {
        "offline": offline,
        "live": live_scores or {},
        "synthetic": bool(meta.get("synthetic")),
    }
    meta["theme_scores"] = payload
    if hasattr(affinity, "metadata"):
        affinity.metadata = meta
    return payload


def run_live_theme_probes(
    bundle: Any,
    affinity_cfg: Any,
    *,
    per_theme: int = 2,
) -> dict[str, Any]:
    """
    Live multi-theme routing probe. Requires a loaded MoE bundle.
    Each theme gets a short AffinityProbe; we store top experts + entropy.
    """
    from aetherforge.affinity.probe import AffinityProbe
    from aetherforge.eval.metrics import routing_entropy
    from aetherforge.groups.forensics import DEFAULT_THEME_BANK

    out: dict[str, Any] = {}
    for tid, spec in DEFAULT_THEME_BANK.items():
        texts = list(spec.get("probes") or [])[:per_theme]
        if not texts:
            continue
        try:
            res = AffinityProbe(bundle, affinity_cfg, domain=tid).run(texts)
            out[tid] = {
                "label": spec.get("label", tid),
                "top": list(res.ranked[:8]),
                "entropy": float(routing_entropy(res.routing_freq)),
                "load_balance_cv": float(res.load_balance_cv or 0.0),
                "probe_tokens": int(res.probe_tokens or 0),
            }
        except Exception as e:
            log.warning("live theme probe %s failed: %s", tid, e)
            out[tid] = {"error": str(e), "label": spec.get("label", tid)}
    return out
