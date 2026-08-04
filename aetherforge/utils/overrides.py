"""Dotlist config overrides: data.domain=logistics training.max_steps=50"""

from __future__ import annotations

from typing import Any


def coerce_value(raw: str) -> Any:
    s = raw.strip()
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "none", "~"):
        return None
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    # int
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        return int(s)
    # float
    try:
        if any(c in s for c in (".", "e", "E")):
            return float(s)
    except ValueError:
        pass
    # JSON-ish list: [a,b] or comma list for simple cases
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [coerce_value(x.strip()) for x in inner.split(",")]
    return s


def parse_overrides(items: list[str]) -> dict[str, Any]:
    """Parse repeated CLI -o key=value into a nested dict."""
    root: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid override (want key=value): {item}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty override key: {item}")
        val = coerce_value(raw)
        cur: dict[str, Any] = root
        parts = key.split(".")
        for part in parts[:-1]:
            nxt = cur.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[part] = nxt
            cur = nxt
        cur[parts[-1]] = val
    return root
