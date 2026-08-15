from aetherforge.data.trajectory_hive import TrajectoryHive
from aetherforge.training.preference import PreferenceAligner, PreferencePair


def test_thd_stub_still_works():
    hive = TrajectoryHive(specialists=["alpha", "beta"], seed=1)
    traj, pairs = hive.generate(["case one"], "demo_field")
    assert traj and pairs
    assert pairs[0]["source"] == "thd"
    assert "chosen" in pairs[0]


def test_thd_live_uses_llm_fn():
    calls = []

    def llm(system: str, user: str) -> str:
        calls.append((system, user))
        who = "alpha" if "alpha" in system else "beta"
        return f"{who} live answer for {user}: assumptions, discriminators, stop rule."

    hive = TrajectoryHive(specialists=["alpha", "beta"], seed=1, llm_fn=llm)
    traj, pairs = hive.generate(["port delay"], "logistics", live=True)
    assert calls
    assert pairs[0]["source"] == "thd_live"
    assert "live answer" in pairs[0]["chosen"] or "live answer" in pairs[0]["rejected"]


def test_thd_live_falls_back_on_llm_error():
    def boom(_s: str, _u: str) -> str:
        raise RuntimeError("down")

    hive = TrajectoryHive(specialists=["alpha"], seed=1, llm_fn=boom)
    traj, pairs = hive.generate(["x"], "demo_field", live=True)
    assert traj
    assert pairs[0]["source"] == "thd_live"
    assert "Specialist" in pairs[0]["chosen"]


def test_aligner_synthesize_live():
    def llm(system: str, user: str) -> str:
        kind = "WIN" if "winning" in system else "LOSE"
        return f"{kind} {user} with a stop condition."

    pairs = PreferenceAligner().synthesize_live(
        ["hard case"], llm_fn=llm, domain="demo_field"
    )
    assert len(pairs) == 1
    assert isinstance(pairs[0], PreferencePair)
    assert pairs[0].source == "thd_live"
    assert "WIN" in pairs[0].chosen
    assert "LOSE" in pairs[0].rejected


def test_aligner_synthesize_skips_identical():
    def llm(_s: str, _u: str) -> str:
        return "same"

    pairs = PreferenceAligner().synthesize_live(["x"], llm_fn=llm)
    assert pairs == []
