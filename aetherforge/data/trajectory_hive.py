"""
Trajectory Hive Distillation (THD) — industry-agnostic.

Specialist names and lenses come from the DomainPack / call site,
never from hard-coded field knowledge.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Optional

from aetherforge.utils.logging import get_logger

log = get_logger("data.thd")


@dataclass
class HiveTrajectory:
    problem: str
    specialist_answers: dict[str, str]
    debate_transcript: str
    consensus: str
    minority: str
    preference_prompt: str


class TrajectoryHive:
    """
    Trajectory Hive Distillation.

    Scaffold mode: multi-view stub answers → preference pairs (always available).
    Live mode: OpenAI-compat `llm_fn(system, user) -> str` per specialist.
    """

    def __init__(
        self,
        specialists: Optional[list[str]] = None,
        seed: int = 42,
        llm_fn: Optional[Any] = None,
    ):
        self.specialists = specialists or ["alpha", "beta", "gamma"]
        self.rng = random.Random(seed)
        self.llm_fn = llm_fn

    def generate(
        self,
        problems: list[str],
        domain: str,
        pairs_per_problem: int = 1,
        *,
        live: bool = False,
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        trajectories: list[dict[str, Any]] = []
        pairs: list[dict[str, str]] = []
        use_live = bool(live and self.llm_fn is not None)

        for problem in problems:
            answers = {}
            for sp in self.specialists:
                if use_live:
                    answers[sp] = self._live_answer(sp, domain, problem)
                else:
                    answers[sp] = self._stub_answer(sp, domain, problem)

            ordered = sorted(answers.items(), key=lambda kv: len(kv[1]), reverse=True)
            best_sp, consensus = ordered[0]
            worst_sp, minority = ordered[-1]

            transcript_lines = [f"Problem: {problem}", f"Domain: {domain}"]
            for sp, ans in answers.items():
                transcript_lines.append(f"[{sp}]: {ans}")
            transcript_lines.append(
                f"[moderator]: Prefer {best_sp} over {worst_sp} for grounding."
            )
            transcript = "\n".join(transcript_lines)

            trajectories.append(
                {
                    "text": transcript + f"\n\n### Final\n{consensus}",
                    "domain": domain,
                    "source": "trajectory_hive",
                    "problem": problem,
                    "winner": best_sp,
                }
            )

            for _ in range(pairs_per_problem):
                pairs.append(
                    {
                        "prompt": f"[{domain}] {problem}\nProvide the best specialist answer.",
                        "chosen": consensus,
                        "rejected": minority,
                        "source": "thd_live" if use_live else "thd",
                        "meta": {
                            "winner": best_sp,
                            "loser": worst_sp,
                            "live": use_live,
                        },
                    }
                )

        log.info(
            "THD: %d problems → %d trajectories, %d preference pairs (live=%s)",
            len(problems),
            len(trajectories),
            len(pairs),
            use_live,
        )
        return trajectories, pairs

    def _live_answer(self, specialist: str, domain: str, problem: str) -> str:
        lens = specialist.replace("_", " ")
        system = (
            f"You are the `{specialist}` specialist in the {domain} hive. "
            f"Answer from a {lens} perspective. Structured bullets. "
            "State assumptions, discriminators, stop/escalate rules, and residual uncertainty. "
            "Do not overclaim."
        )
        try:
            text = self.llm_fn(system, problem)  # type: ignore[misc]
            if text and str(text).strip():
                return str(text).strip()
        except Exception as e:
            log.warning("live THD %s failed (%s); falling back to stub", specialist, e)
        return self._stub_answer(specialist, domain, problem)

    def _stub_answer(self, specialist: str, domain: str, problem: str) -> str:
        depth = 3 + self.rng.randint(0, 4)
        # Generic lenses derived from specialist name — no field ontology
        lens = f"{specialist.replace('_', ' ')} perspective"
        angles = [
            f"Pretest / base-rate framing for: {problem[:80]}",
            f"Discriminators that change next action ({lens})",
            "What would falsify the leading hypothesis",
            "Monitoring plan with explicit stop/escalate rules",
            "Residual uncertainty to disclose to decision owners",
            f"Domain constraints for {domain}",
        ]
        self.rng.shuffle(angles)
        bullets = "\n".join(f"- {angles[i % len(angles)]}" for i in range(depth))
        conf = 0.55 + self.rng.random() * 0.4
        return (
            f"Specialist `{specialist}` ({lens}) — domain {domain}:\n"
            f"{bullets}\n"
            f"Recommendation: act under {lens}; confidence ≈ {conf:.2f}; "
            f"revisit if new evidence contradicts the frame.\n"
            f"Problem restated: {problem[:160]}"
        )
