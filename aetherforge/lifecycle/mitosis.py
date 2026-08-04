"""Expert mitosis helpers — conceptual split of overloaded experts."""

from __future__ import annotations

from typing import Any


def mitosis_spec(layer: int, expert: int, child_count: int = 2) -> dict[str, Any]:
    """
    Describe a mitosis operation for export / future weight ops.

    Live application requires architecture-specific expert ModuleList surgery
    and router logit expansion — intentionally deferred to family adapters.
    """
    return {
        "op": "mitosis",
        "parent": {"layer": layer, "expert": expert},
        "children": [
            {"layer": layer, "expert": f"{expert}_child_{i}", "init": "parent_copy+noise"}
            for i in range(child_count)
        ],
        "router": "expand_logits_and_split_mass",
        "status": "planned_not_applied",
    }
