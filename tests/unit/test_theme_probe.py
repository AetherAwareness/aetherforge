from aetherforge.affinity.theme_probe import (
    attach_theme_scores,
    collect_theme_probes,
    score_themes_offline,
)
from aetherforge.data.domain_pack import DomainPack


class _Aff:
    def __init__(self):
        self.domain = "logistics"
        self.metadata = {"synthetic": True}


def test_offline_theme_scores_use_pack_keywords():
    pack = DomainPack(
        domain="logistics",
        keywords=["inventory", "warehouse", "api"],
        topics=["Write a python function for warehouse routing."],
    )
    scores = score_themes_offline("logistics", pack=pack)
    assert scores
    assert scores.get("code_software", 0) > 0


def test_attach_writes_metadata():
    aff = _Aff()
    payload = attach_theme_scores(aff, pack=DomainPack(domain="demo_field"))
    assert "offline" in payload
    assert aff.metadata["theme_scores"]["synthetic"] is True


def test_collect_includes_pack_topics():
    pack = DomainPack(domain="energy", topics=["nodal price spike"])
    items = collect_theme_probes(pack)
    assert any(i.get("text") == "nodal price spike" for i in items)
    assert any(i.get("theme_id", "").startswith("pack:") for i in items)
