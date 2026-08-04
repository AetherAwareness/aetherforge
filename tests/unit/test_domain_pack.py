from pathlib import Path

from aetherforge.data.domain_pack import DomainPack, resolve_domain_pack, load_domain_pack
from aetherforge.data.synthetic import generate_self_instruct
from aetherforge.utils.config import DataConfig, SyntheticConfig


ROOT = Path(__file__).resolve().parents[2]


def test_resolve_generic_no_industry_bleed():
    pack = resolve_domain_pack(DataConfig(domain="aerospace_composites"))
    assert pack.domain == "aerospace_composites"
    assert pack.topics
    # keywords derived from slug/topics — not a medical table
    joined = " ".join(pack.keywords).lower()
    assert "troponin" not in joined
    assert "cancer" not in joined
    assert "aerospace" in joined or "composites" in joined


def test_inline_topics_override():
    pack = resolve_domain_pack(
        DataConfig(
            domain="retail_pricing",
            topics=["promo elasticity under stockout"],
            keywords=["promo", "elasticity", "stockout"],
            actions=["Re-estimate demand curve with censored sales."],
        )
    )
    assert pack.topics == ["promo elasticity under stockout"]
    assert "promo" in pack.keywords


def test_load_example_pack_via_domain_config():
    # example logistics config is a domain recipe; resolve via DataConfig fields
    from aetherforge.utils.config import load_config

    cfg = load_config(
        ROOT / "configs" / "base.yaml",
        ROOT / "configs" / "domains" / "example_logistics.yaml",
    )
    pack = resolve_domain_pack(cfg.data)
    assert pack.domain == "logistics"
    assert any("inventory" in t for t in pack.topics)


def test_synthetic_uses_pack_not_hardcode():
    pack = DomainPack(
        domain="maritime_insurance",
        topics=["hull claim under piracy rider"],
        keywords=["hull", "piracy", "rider", "claim"],
        actions=["Check war-risk clause effective dates."],
        populations=["underwriter"],
        contexts=["claims desk"],
        angles=["coverage"],
        hints=["prefer policy language"],
    )
    recs = generate_self_instruct(
        pack.domain,
        SyntheticConfig(enabled=True, num_samples=5),
        seed=1,
        pack=pack,
    )
    assert len(recs) == 5
    blob = " ".join(r["text"] for r in recs).lower()
    assert "maritime_insurance" in blob or "hull" in blob
    assert "troponin" not in blob
    assert "checkpoint inhibitor" not in blob
